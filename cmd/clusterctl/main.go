// clusterctl — cliente de línea de comandos del coordinador (PC4).
//
// Sirve para verificar el clúster y entrenar de forma distribuida sin la API:
//
//	go run ./cmd/clusterctl -nodes localhost:9001,localhost:9002,localhost:9003 status
//	go run ./cmd/clusterctl -nodes ... train            # RF + regresión logística + evaluación
//	go run ./cmd/clusterctl -nodes ... -trees 60 -depth 14 -epochs 4 train
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/cluster"
)

func main() {
	nodes := flag.String("nodes", "localhost:9001,localhost:9002,localhost:9003", "direcciones de los nodos")
	trees := flag.Int("trees", 60, "árboles del Random Forest")
	depth := flag.Int("depth", 14, "profundidad máxima")
	sample := flag.Int("sample", 250000, "filas bootstrap por árbol")
	epochs := flag.Int("epochs", 4, "épocas de la regresión logística")
	batch := flag.Int("batch", 2048, "mini-batch por nodo y paso")
	wait := flag.Duration("wait", 2*time.Minute, "espera máxima a que los nodos estén listos")
	out := flag.String("out", ".", "carpeta base para models/ y reports/")
	flag.Parse()
	cmd := flag.Arg(0)
	if cmd == "" {
		cmd = "status"
	}

	c := cluster.NewCluster(strings.Split(*nodes, ","))
	if err := c.Connect(context.Background(), *wait); err != nil {
		log.Fatal(err)
	}
	switch cmd {
	case "status":
		b, _ := json.MarshalIndent(c.Status(), "", "  ")
		fmt.Println(string(b))
	case "train":
		train(c, cluster.TrainParams{Trees: *trees, Depth: *depth, Sample: *sample, Epochs: *epochs, Batch: *batch}, *out)
	default:
		log.Fatalf("comando desconocido %q (usa status | train)", cmd)
	}
}

func train(c *cluster.Cluster, p cluster.TrainParams, out string) {
	res, err := c.TrainAll(p, log.Printf)
	if err != nil {
		log.Fatal(err)
	}
	if err := res.SaveModels(out + "/models"); err != nil {
		log.Fatal(err)
	}
	if err := res.SaveReport(out + "/reports/cluster_train_report.json"); err != nil {
		log.Fatal(err)
	}
	log.Printf("listo en %.1fs → %s/models y %s/reports/cluster_train_report.json", res.Total, out, out)
}
