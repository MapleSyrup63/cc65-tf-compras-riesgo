// node — nodo de cómputo ML del clúster (PC4).
//
// Carga el dataset, prepara su partición/bins (idénticos en todos los nodos) y atiende
// por TCP las tareas del coordinador: entrenar árboles, calcular gradientes, puntuar.
//
// Uso:
//
//	go run ./cmd/node -id nodo1 -listen :9001 -data data/processed/dataset.bin
package main

import (
	"context"
	"flag"
	"log"
	"net"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/cluster"
	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/ml"
)

func main() {
	id := flag.String("id", "nodo1", "identificador del nodo")
	listen := flag.String("listen", ":9001", "dirección TCP donde escucha")
	data := flag.String("data", "data/processed/dataset.bin", "dataset binario generado por el loader")
	local := flag.Int("local-workers", max(runtime.NumCPU()/3, 1), "goroutines internas del nodo")
	flag.Parse()

	t0 := time.Now()
	d, err := ml.LoadDataset(*data)
	if err != nil {
		log.Fatalf("[%s] no se pudo cargar %s: %v", *id, *data, err)
	}
	log.Printf("[%s] dataset N=%d D=%d cargado en %.1fs", *id, d.N, d.D, time.Since(t0).Seconds())

	t1 := time.Now()
	node := cluster.NewNode(*id, d, *local)
	log.Printf("[%s] listo (bins + shards) en %.1fs | goroutines locales=%d | CPU=%d",
		*id, time.Since(t1).Seconds(), *local, runtime.NumCPU())

	ln, err := net.Listen("tcp", *listen)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("[%s] escuchando en %s", *id, ln.Addr())

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := node.Serve(ctx, ln); err != nil {
		log.Fatal(err)
	}
	log.Printf("[%s] detenido", *id)
}
