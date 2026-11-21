// apibench — prueba de carga de /predict para evidenciar la latencia (< 100 ms).
//
//	go run ./cmd/apibench -url http://localhost:8080 -n 5000 -c 32
//
// Mide el tiempo de ida y vuelta visto por el cliente (incluye red y JSON) y también
// el X-Latency-Ms informado por el servidor. Guarda reports/api_latency.json.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"sort"
	"strconv"
	"sync"
	"time"
)

func main() {
	url := flag.String("url", "http://localhost:8080", "URL base de la API")
	user := flag.String("user", "analista", "usuario")
	pass := flag.String("pass", "analista123", "contraseña")
	n := flag.Int("n", 5000, "peticiones totales")
	c := flag.Int("c", 32, "clientes concurrentes")
	out := flag.String("out", "reports/api_latency.json", "archivo de resultados")
	flag.Parse()

	cli := &http.Client{Timeout: 10 * time.Second, Transport: &http.Transport{MaxIdleConnsPerHost: *c, MaxConnsPerHost: *c}}
	lb, _ := json.Marshal(map[string]string{"usuario": *user, "password": *pass})
	resp, err := cli.Post(*url+"/api/v1/auth/login", "application/json", bytes.NewReader(lb))
	if err != nil {
		log.Fatal(err)
	}
	var lr struct {
		Token string `json:"token"`
	}
	json.NewDecoder(resp.Body).Decode(&lr)
	resp.Body.Close()
	if lr.Token == "" {
		log.Fatalf("login falló (HTTP %d)", resp.StatusCode)
	}

	call := func(i int) (client, server float64, status int) {
		// 50 % de consultas repetidas (aciertan la caché) y 50 % nuevas
		monto := 500 + (i%50)*900
		if i%2 == 1 {
			monto = 500 + (i%5000)*37
		}
		body, _ := json.Marshal(map[string]any{"monto_pen": monto, "features": map[string]float64{"tc_hasta_8uit": float64(i % 2)}})
		req, _ := http.NewRequest("POST", *url+"/api/v1/predict", bytes.NewReader(body))
		req.Header.Set("Authorization", "Bearer "+lr.Token)
		req.Header.Set("Content-Type", "application/json")
		t0 := time.Now()
		r, err := cli.Do(req)
		if err != nil {
			return 0, 0, 0
		}
		defer r.Body.Close()
		var tmp map[string]any
		json.NewDecoder(r.Body).Decode(&tmp)
		client = float64(time.Since(t0).Microseconds()) / 1000
		server, _ = strconv.ParseFloat(r.Header.Get("X-Latency-Ms"), 64)
		return client, server, r.StatusCode
	}

	for i := 0; i < 200; i++ { // calentamiento
		call(i)
	}
	lat := make([]float64, *n)
	srv := make([]float64, *n)
	var errs, bad int
	var mu sync.Mutex
	var wg sync.WaitGroup
	idx := make(chan int, *n)
	for i := 0; i < *n; i++ {
		idx <- i
	}
	close(idx)
	t0 := time.Now()
	for w := 0; w < *c; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range idx {
				cl, sv, st := call(i)
				lat[i], srv[i] = cl, sv // cada petición escribe SOLO su posición
				if st == 0 || st != 200 {
					mu.Lock()
					if st == 0 {
						errs++
					} else {
						bad++
					}
					mu.Unlock()
				}
			}
		}()
	}
	wg.Wait()
	dur := time.Since(t0).Seconds()

	q := func(a []float64, p float64) float64 {
		b := append([]float64(nil), a...)
		sort.Float64s(b)
		return b[int(p*float64(len(b)-1))]
	}
	mean := func(a []float64) float64 {
		var s float64
		for _, v := range a {
			s += v
		}
		return s / float64(len(a))
	}
	res := map[string]any{
		"peticiones": *n, "concurrencia": *c, "segundos": dur, "req_por_seg": float64(*n) / dur,
		"errores_red": errs, "respuestas_no_200": bad,
		"cliente_ms":             map[string]float64{"media": mean(lat), "p50": q(lat, .5), "p95": q(lat, .95), "p99": q(lat, .99), "max": q(lat, 1)},
		"servidor_ms":            map[string]float64{"media": mean(srv), "p50": q(srv, .5), "p95": q(srv, .95), "p99": q(srv, .99), "max": q(srv, 1)},
		"cumple_menor_100ms_p99": q(lat, .99) < 100 && errs == 0 && bad == 0,
	}
	b, _ := json.MarshalIndent(res, "", "  ")
	fmt.Println(string(b))
	if err := os.WriteFile(*out, b, 0o644); err != nil {
		log.Printf("no se pudo escribir %s: %v", *out, err)
	}
}
