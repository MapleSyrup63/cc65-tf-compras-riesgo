// seed — carga la muestra de órdenes (data/sample/ordenes_muestra.csv) en MongoDB
// (colección ordenes_muestra). Es idempotente: upsert por código de orden.
//
//	go run ./cmd/seed -csv data/sample/ordenes_muestra.csv -mongo mongodb://localhost:27017
package main

import (
	"context"
	"encoding/csv"
	"flag"
	"io"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/mongostore"
	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/store"
)

func main() {
	path := flag.String("csv", "data/sample/ordenes_muestra.csv", "CSV de la muestra")
	uri := flag.String("mongo", "mongodb://localhost:27017", "URI de MongoDB")
	db := flag.String("db", "riesgogo", "base de datos")
	flag.Parse()

	f, err := os.Open(*path)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	rd := csv.NewReader(f)
	rd.FieldsPerRecord = -1
	hdr, err := rd.Read()
	if err != nil {
		log.Fatal(err)
	}
	col := map[string]int{}
	for i, h := range hdr {
		col[strings.TrimSpace(h)] = i
	}
	for _, need := range []string{"ruc_entidad", "orden", "entidad", "departamento", "monto_pen", "riesgo"} {
		if _, ok := col[need]; !ok {
			log.Fatalf("falta la columna %q en %s", need, *path)
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	repo, err := mongostore.New(ctx, *uri, *db)
	if err != nil {
		log.Fatal(err)
	}
	defer repo.Close(context.Background())

	get := func(rec []string, k string) string {
		if i, ok := col[k]; ok && i < len(rec) {
			return rec[i]
		}
		return ""
	}
	flag01 := func(s string) bool { return s == "1" || strings.EqualFold(s, "true") }
	var batch []store.OrdenMuestra
	total := 0
	flush := func() {
		if len(batch) == 0 {
			return
		}
		if err := repo.InsertOrdenes(ctx, batch); err != nil {
			log.Fatal(err)
		}
		total += len(batch)
		batch = batch[:0]
	}
	for {
		rec, err := rd.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		monto, _ := strconv.ParseFloat(get(rec, "monto_pen"), 64)
		batch = append(batch, store.OrdenMuestra{
			ID: get(rec, "ruc_entidad") + "|" + get(rec, "orden"), Entidad: get(rec, "entidad"),
			Departamento: get(rec, "departamento"), NivelGobierno: get(rec, "nivel_gobierno"), Rubro: get(rec, "rubro"),
			Objeto: get(rec, "objeto"), TipoContratacion: get(rec, "tipo_contratacion"), MontoPEN: monto,
			Anulada: flag01(get(rec, "anulada")), Sobrecosto: flag01(get(rec, "sobrecosto")),
			Riesgo: flag01(get(rec, "riesgo")), Split: get(rec, "split")})
		if len(batch) >= 1000 {
			flush()
		}
	}
	flush()
	c, _ := repo.Conteos(ctx)
	log.Printf("órdenes cargadas: %d | colecciones: %v", total, c)
}
