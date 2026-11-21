package cluster

import (
	"bufio"
	"context"
	"encoding/gob"
	"errors"
	"fmt"
	"net"
	"sync"
	"sync/atomic"
	"time"
)

// NodeConn es la conexión TCP persistente del coordinador con un nodo.
// El mutex serializa las llamadas sobre la MISMA conexión (gob es un flujo único);
// las llamadas a nodos distintos corren en paralelo.
type NodeConn struct {
	Addr     string
	healthy  atomic.Bool
	lastFail atomic.Int64 // instante (UnixNano) del último fallo de conexión

	infoMu  sync.RWMutex // protege info (se escribe en Connect, se lee desde las goroutines de la API)
	info    HelloResp
	infoSet bool

	mu   sync.Mutex
	conn net.Conn
	enc  *gob.Encoder
	bw   *bufio.Writer
	dec  *gob.Decoder
}

const (
	dialTimeout = 800 * time.Millisecond // plazo para abrir la conexión TCP con un nodo
	failFast    = 2 * time.Second        // tras un fallo, no se reintenta la conexión durante este tiempo
)

func (n *NodeConn) Healthy() bool { return n.healthy.Load() }

// Info devuelve los metadatos del dataset que informó el nodo (ok=false si aún no respondió).
func (n *NodeConn) Info() (HelloResp, bool) {
	n.infoMu.RLock()
	defer n.infoMu.RUnlock()
	return n.info, n.infoSet
}

func (n *NodeConn) setInfo(h HelloResp) {
	n.infoMu.Lock()
	n.info, n.infoSet = h, true
	n.infoMu.Unlock()
}

func (n *NodeConn) dial(timeout time.Duration) error {
	c, err := net.DialTimeout("tcp", n.Addr, timeout)
	if err != nil {
		return err
	}
	if tc, ok := c.(*net.TCPConn); ok {
		tc.SetNoDelay(true)
		tc.SetKeepAlive(true)
	}
	n.conn = c
	n.bw = bufio.NewWriterSize(c, 1<<20)
	n.enc = gob.NewEncoder(n.bw)
	n.dec = gob.NewDecoder(bufio.NewReaderSize(c, 1<<20))
	return nil
}

func (n *NodeConn) closeLocked() {
	if n.conn != nil {
		n.conn.Close()
		n.conn = nil
	}
}

// Call envía una solicitud y espera la respuesta con un plazo máximo. Si la conexión
// falla la cierra y marca el nodo como caído; el siguiente Call intenta reconectar.
func (n *NodeConn) Call(req *Request, timeout time.Duration) (*Response, error) {
	n.mu.Lock()
	defer n.mu.Unlock()
	if n.conn == nil {
		// Nodo caído hace poco: falla rápido en vez de esperar otro plazo de conexión,
		// para que /cluster y el reparto de tareas no se bloqueen por un nodo muerto.
		if since := time.Since(time.Unix(0, n.lastFail.Load())); since < failFast {
			return nil, fmt.Errorf("%s: nodo caído (reintento en %s)", n.Addr, (failFast - since).Round(time.Second))
		}
		if err := n.dial(dialTimeout); err != nil {
			n.healthy.Store(false)
			n.lastFail.Store(time.Now().UnixNano())
			return nil, fmt.Errorf("%s: %w", n.Addr, err)
		}
	}
	n.conn.SetDeadline(time.Now().Add(timeout))
	var resp Response
	err := n.enc.Encode(req)
	if err == nil {
		err = n.bw.Flush()
	}
	if err == nil {
		err = n.dec.Decode(&resp)
	}
	if err != nil {
		n.closeLocked()
		n.healthy.Store(false)
		n.lastFail.Store(time.Now().UnixNano())
		return nil, fmt.Errorf("%s: %w", n.Addr, err)
	}
	n.healthy.Store(true)
	if resp.Err != "" {
		return nil, fmt.Errorf("%s: %s", n.Addr, resp.Err)
	}
	return &resp, nil
}

// Cluster es el coordinador: conoce los nodos y reparte tareas entre ellos.
type Cluster struct {
	Nodes []*NodeConn
	// Timeout por llamada (entrenar árboles puede tardar varios segundos).
	Timeout time.Duration
}

func NewCluster(addrs []string) *Cluster {
	c := &Cluster{Timeout: 10 * time.Minute}
	for _, a := range addrs {
		c.Nodes = append(c.Nodes, &NodeConn{Addr: a})
	}
	return c
}

// Connect espera a que los nodos estén listos (cargan el dataset al arrancar) y
// obtiene su OpHello. Reintenta con espera hasta `wait`. Falla si ninguno responde.
func (c *Cluster) Connect(ctx context.Context, wait time.Duration) error {
	deadline := time.Now().Add(wait)
	var wg sync.WaitGroup
	for _, n := range c.Nodes {
		wg.Add(1)
		go func(n *NodeConn) {
			defer wg.Done()
			for {
				resp, err := n.Call(&Request{Op: OpHello}, 5*time.Second)
				if err == nil {
					n.setInfo(*resp.Hello)
					return
				}
				if time.Now().After(deadline) || ctx.Err() != nil {
					return
				}
				select {
				case <-ctx.Done():
					return
				case <-time.After(time.Second):
				}
			}
		}(n)
	}
	wg.Wait()
	if len(c.Healthy()) == 0 {
		return errors.New("ningún nodo respondió")
	}
	return nil
}

// Healthy lista los nodos vivos.
func (c *Cluster) Healthy() []*NodeConn {
	var out []*NodeConn
	for _, n := range c.Nodes {
		if n.Healthy() {
			out = append(out, n)
		}
	}
	return out
}

// info devuelve los metadatos del dataset (idénticos en todos los nodos).
func (c *Cluster) info() (HelloResp, error) {
	for _, n := range c.Nodes {
		if h, ok := n.Info(); ok {
			return h, nil
		}
	}
	return HelloResp{}, errors.New("ningún nodo ha informado su dataset todavía")
}

// Info expone los metadatos del dataset del clúster (para la API y clusterctl).
func (c *Cluster) Info() (HelloResp, error) { return c.info() }

// NodeStatus es el estado de un nodo para /cluster.
type NodeStatus struct {
	Addr      string     `json:"addr"`
	Healthy   bool       `json:"healthy"`
	LatencyMS float64    `json:"latency_ms"`
	Stats     *StatsResp `json:"stats,omitempty"`
	Error     string     `json:"error,omitempty"`
}

// Status consulta a todos los nodos en paralelo (latido + métricas).
func (c *Cluster) Status() []NodeStatus {
	out := make([]NodeStatus, len(c.Nodes))
	var wg sync.WaitGroup
	for i, n := range c.Nodes {
		wg.Add(1)
		go func(i int, n *NodeConn) {
			defer wg.Done()
			t0 := time.Now()
			resp, err := n.Call(&Request{Op: OpStats}, 3*time.Second)
			st := NodeStatus{Addr: n.Addr, LatencyMS: float64(time.Since(t0).Microseconds()) / 1000}
			if err != nil {
				st.Error = err.Error()
			} else {
				st.Healthy, st.Stats = true, resp.Stats
			}
			out[i] = st // cada goroutine escribe SOLO su posición
		}(i, n)
	}
	wg.Wait()
	return out
}

// exec ejecuta una tarea en su nodo preferido (task % N). Si ese nodo falla, la
// reasigna al siguiente: como todos los nodos tienen el dataset completo y derivan los
// mismos shards, cualquiera puede ejecutar cualquier tarea (tolerancia a fallos).
func (c *Cluster) exec(task int, req *Request) (resp *Response, node string, retries int, err error) {
	N := len(c.Nodes)
	if N == 0 {
		return nil, "", 0, errors.New("clúster vacío")
	}
	var last error
	for i := 0; i < N; i++ {
		n := c.Nodes[(task+i)%N]
		r, e := n.Call(req, c.Timeout)
		if e == nil {
			return r, n.Addr, i, nil
		}
		last = e
	}
	return nil, "", N, fmt.Errorf("tarea %d falló en todos los nodos: %w", task, last)
}
