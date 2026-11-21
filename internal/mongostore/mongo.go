// Package mongostore implementa store.Repo sobre MongoDB (driver oficial v2).
//
// Colecciones: usuarios, entrenamientos, predicciones, metricas_nodos, ordenes_muestra.
package mongostore

import (
	"context"
	"errors"
	"time"

	"go.mongodb.org/mongo-driver/v2/bson"
	"go.mongodb.org/mongo-driver/v2/mongo"
	"go.mongodb.org/mongo-driver/v2/mongo/options"

	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/store"
)

// Repo es el adaptador de MongoDB.
type Repo struct {
	client *mongo.Client
	db     *mongo.Database
}

// Nombres de colección.
const (
	colUsuarios = "usuarios"
	colEntren   = "entrenamientos"
	colPred     = "predicciones"
	colMetricas = "metricas_nodos"
	colOrdenes  = "ordenes_muestra"
)

// New conecta, verifica con ping y crea los índices. uri p.ej. mongodb://mongodb:27017
func New(ctx context.Context, uri, dbName string) (*Repo, error) {
	client, err := mongo.Connect(options.Client().ApplyURI(uri).SetServerSelectionTimeout(5 * time.Second))
	if err != nil {
		return nil, err
	}
	r := &Repo{client: client, db: client.Database(dbName)}
	if err := r.Ping(ctx); err != nil {
		client.Disconnect(context.Background())
		return nil, err
	}
	if err := r.ensureIndexes(ctx); err != nil {
		client.Disconnect(context.Background())
		return nil, err
	}
	return r, nil
}

func (r *Repo) ensureIndexes(ctx context.Context) error {
	idx := []struct {
		col   string
		model mongo.IndexModel
	}{
		{colPred, mongo.IndexModel{Keys: bson.D{{Key: "ts", Value: -1}}}},
		{colPred, mongo.IndexModel{Keys: bson.D{{Key: "usuario", Value: 1}, {Key: "ts", Value: -1}}}},
		{colEntren, mongo.IndexModel{Keys: bson.D{{Key: "inicio", Value: -1}}}},
		{colMetricas, mongo.IndexModel{Keys: bson.D{{Key: "ts", Value: -1}},
			Options: options.Index().SetExpireAfterSeconds(7 * 24 * 3600)}}, // las métricas expiran a los 7 días
		{colOrdenes, mongo.IndexModel{Keys: bson.D{{Key: "departamento", Value: 1}, {Key: "riesgo", Value: 1}}}},
	}
	for _, i := range idx {
		if _, err := r.db.Collection(i.col).Indexes().CreateOne(ctx, i.model); err != nil {
			return err
		}
	}
	return nil
}

func (r *Repo) Ping(ctx context.Context) error  { return r.client.Ping(ctx, nil) }
func (r *Repo) Close(ctx context.Context) error { return r.client.Disconnect(ctx) }

func (r *Repo) UpsertUsuario(ctx context.Context, u store.Usuario) error {
	_, err := r.db.Collection(colUsuarios).ReplaceOne(ctx, bson.M{"_id": u.Usuario}, u, options.Replace().SetUpsert(true))
	return err
}

func (r *Repo) GetUsuario(ctx context.Context, n string) (*store.Usuario, error) {
	var u store.Usuario
	err := r.db.Collection(colUsuarios).FindOne(ctx, bson.M{"_id": n}).Decode(&u)
	if errors.Is(err, mongo.ErrNoDocuments) {
		return nil, store.ErrNotFound
	}
	return &u, err
}

func (r *Repo) SaveEntrenamiento(ctx context.Context, e store.Entrenamiento) error {
	_, err := r.db.Collection(colEntren).ReplaceOne(ctx, bson.M{"_id": e.ID}, e, options.Replace().SetUpsert(true))
	return err
}

func (r *Repo) GetEntrenamiento(ctx context.Context, id string) (*store.Entrenamiento, error) {
	var e store.Entrenamiento
	err := r.db.Collection(colEntren).FindOne(ctx, bson.M{"_id": id}).Decode(&e)
	if errors.Is(err, mongo.ErrNoDocuments) {
		return nil, store.ErrNotFound
	}
	return &e, err
}

func (r *Repo) ListEntrenamientos(ctx context.Context, limit int) ([]store.Entrenamiento, error) {
	cur, err := r.db.Collection(colEntren).Find(ctx, bson.M{},
		options.Find().SetSort(bson.D{{Key: "inicio", Value: -1}}).SetLimit(int64(limit)))
	if err != nil {
		return nil, err
	}
	var out []store.Entrenamiento
	return out, cur.All(ctx, &out)
}

func (r *Repo) InsertPredicciones(ctx context.Context, p []store.Prediccion) error {
	if len(p) == 0 {
		return nil
	}
	docs := make([]any, len(p))
	for i := range p {
		docs[i] = p[i]
	}
	_, err := r.db.Collection(colPred).InsertMany(ctx, docs)
	return err
}

func (r *Repo) ListPredicciones(ctx context.Context, limit int, usuario string) ([]store.Prediccion, error) {
	filter := bson.M{}
	if usuario != "" {
		filter["usuario"] = usuario
	}
	cur, err := r.db.Collection(colPred).Find(ctx, filter,
		options.Find().SetSort(bson.D{{Key: "ts", Value: -1}}).SetLimit(int64(limit)))
	if err != nil {
		return nil, err
	}
	var out []store.Prediccion
	return out, cur.All(ctx, &out)
}

func (r *Repo) InsertMetricasNodos(ctx context.Context, m []store.MetricaNodo) error {
	if len(m) == 0 {
		return nil
	}
	docs := make([]any, len(m))
	for i := range m {
		docs[i] = m[i]
	}
	_, err := r.db.Collection(colMetricas).InsertMany(ctx, docs)
	return err
}

func (r *Repo) ListMetricasNodos(ctx context.Context, limit int) ([]store.MetricaNodo, error) {
	cur, err := r.db.Collection(colMetricas).Find(ctx, bson.M{},
		options.Find().SetSort(bson.D{{Key: "ts", Value: -1}}).SetLimit(int64(limit)))
	if err != nil {
		return nil, err
	}
	var out []store.MetricaNodo
	return out, cur.All(ctx, &out)
}

// InsertOrdenes hace upsert por código de orden (la carga se puede repetir sin duplicar).
func (r *Repo) InsertOrdenes(ctx context.Context, o []store.OrdenMuestra) error {
	if len(o) == 0 {
		return nil
	}
	models := make([]mongo.WriteModel, len(o))
	for i := range o {
		models[i] = mongo.NewReplaceOneModel().SetFilter(bson.M{"_id": o[i].ID}).SetReplacement(o[i]).SetUpsert(true)
	}
	_, err := r.db.Collection(colOrdenes).BulkWrite(ctx, models, options.BulkWrite().SetOrdered(false))
	return err
}

func (r *Repo) ListOrdenes(ctx context.Context, f store.OrdenFiltro) ([]store.OrdenMuestra, int64, error) {
	filter := bson.M{}
	if f.Departamento != "" {
		filter["departamento"] = f.Departamento
	}
	if f.Riesgo != nil {
		filter["riesgo"] = *f.Riesgo
	}
	col := r.db.Collection(colOrdenes)
	total, err := col.CountDocuments(ctx, filter)
	if err != nil {
		return nil, 0, err
	}
	cur, err := col.Find(ctx, filter, options.Find().SetSort(bson.D{{Key: "_id", Value: 1}}).
		SetSkip(int64(f.Skip)).SetLimit(int64(f.Limit)))
	if err != nil {
		return nil, 0, err
	}
	var out []store.OrdenMuestra
	return out, total, cur.All(ctx, &out)
}

func (r *Repo) Conteos(ctx context.Context) (map[string]int64, error) {
	out := map[string]int64{}
	for _, c := range []string{colUsuarios, colEntren, colPred, colMetricas, colOrdenes} {
		n, err := r.db.Collection(c).EstimatedDocumentCount(ctx)
		if err != nil {
			return nil, err
		}
		out[c] = n
	}
	return out, nil
}
