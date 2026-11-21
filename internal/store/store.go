// Package store define las ENTIDADES persistentes de RiesgoGo y los contratos de las
// dos capas de almacenamiento de la PC4:
//
//	Repo  → persistente (MongoDB): usuarios, entrenamientos, predicciones, métricas de
//	        nodos y una muestra de órdenes para consulta.
//	Cache → caché/colaborativo (Redis): predicciones precalculadas y estado del clúster.
//
// La API depende solo de estas interfaces; hay implementaciones en memoria (pruebas y
// modo degradado) y adaptadores para MongoDB (internal/mongostore) y Redis (internal/redisstore).
package store

import (
	"context"
	"errors"
	"time"
)

// ErrNotFound se devuelve cuando un documento no existe.
var ErrNotFound = errors.New("no encontrado")

// Usuario: credenciales de la API. La contraseña NUNCA se guarda: solo PBKDF2-SHA256 + sal.
type Usuario struct {
	Usuario string    `json:"usuario" bson:"_id"`
	Salt    string    `json:"-" bson:"salt"`
	Hash    string    `json:"-" bson:"hash"`
	Rol     string    `json:"rol" bson:"rol"` // "admin" | "analista"
	Creado  time.Time `json:"creado" bson:"creado"`
}

// TareaNodo: trabajo ejecutado por un nodo durante un entrenamiento (evidencia del reparto).
type TareaNodo struct {
	Tipo           string  `json:"tipo" bson:"tipo"` // "rf" | "lr"
	Tarea          int     `json:"tarea" bson:"tarea"`
	Nodo           string  `json:"nodo" bson:"nodo"`
	Items          int     `json:"items" bson:"items"`
	SegundosNodo   float64 `json:"segundos_nodo" bson:"segundos_nodo"`
	SegundosTotal  float64 `json:"segundos_total" bson:"segundos_total"`
	Reasignaciones int     `json:"reasignaciones" bson:"reasignaciones"`
}

// Entrenamiento: una corrida de entrenamiento distribuido.
type Entrenamiento struct {
	ID         string         `json:"id" bson:"_id"`
	Estado     string         `json:"estado" bson:"estado"` // en_curso | completado | error
	Parametros map[string]any `json:"parametros" bson:"parametros"`
	Usuario    string         `json:"usuario" bson:"usuario"`
	Inicio     time.Time      `json:"inicio" bson:"inicio"`
	Fin        *time.Time     `json:"fin,omitempty" bson:"fin,omitempty"`
	Segundos   float64        `json:"segundos" bson:"segundos"`
	Nodos      int            `json:"nodos" bson:"nodos"`
	AUCRF      float64        `json:"auc_rf" bson:"auc_rf"`
	AUCLR      float64        `json:"auc_lr" bson:"auc_lr"`
	Error      string         `json:"error,omitempty" bson:"error,omitempty"`
	Log        []string       `json:"log" bson:"log"`
	Tareas     []TareaNodo    `json:"tareas,omitempty" bson:"tareas,omitempty"`
}

// Prediccion: una consulta a /predict y su resultado.
type Prediccion struct {
	ID         string             `json:"id" bson:"_id"`
	TS         time.Time          `json:"ts" bson:"ts"`
	Usuario    string             `json:"usuario" bson:"usuario"`
	Entrada    map[string]float64 `json:"entrada" bson:"entrada"`
	RF         float64            `json:"score_rf" bson:"score_rf"`
	LR         float64            `json:"score_lr" bson:"score_lr"`
	Riesgo     bool               `json:"riesgo" bson:"riesgo"`
	Nivel      string             `json:"nivel" bson:"nivel"`
	LatenciaMS float64            `json:"latencia_ms" bson:"latencia_ms"`
	Cache      bool               `json:"cache" bson:"cache"`
}

// MetricaNodo: muestra periódica del estado de un nodo (panel de administración).
type MetricaNodo struct {
	ID         string    `json:"id" bson:"_id"`
	TS         time.Time `json:"ts" bson:"ts"`
	Nodo       string    `json:"nodo" bson:"nodo"`
	Sano       bool      `json:"sano" bson:"sano"`
	LatenciaMS float64   `json:"latencia_ms" bson:"latencia_ms"`
	Requests   int64     `json:"requests" bson:"requests"`
	BusySeg    float64   `json:"busy_seg" bson:"busy_seg"`
	HeapMB     float64   `json:"heap_mb" bson:"heap_mb"`
	Goroutines int       `json:"goroutines" bson:"goroutines"`
}

// OrdenMuestra: orden de compra real de la muestra de 20,000 filas del repositorio.
type OrdenMuestra struct {
	ID               string  `json:"id" bson:"_id"` // código de la orden
	Entidad          string  `json:"entidad" bson:"entidad"`
	Departamento     string  `json:"departamento" bson:"departamento"`
	NivelGobierno    string  `json:"nivel_gobierno" bson:"nivel_gobierno"`
	Rubro            string  `json:"rubro" bson:"rubro"`
	Objeto           string  `json:"objeto" bson:"objeto"`
	TipoContratacion string  `json:"tipo_contratacion" bson:"tipo_contratacion"`
	MontoPEN         float64 `json:"monto_pen" bson:"monto_pen"`
	Anulada          bool    `json:"anulada" bson:"anulada"`
	Sobrecosto       bool    `json:"sobrecosto" bson:"sobrecosto"`
	Riesgo           bool    `json:"riesgo" bson:"riesgo"`
	Split            string  `json:"split" bson:"split"`
}

// OrdenFiltro filtros del listado de órdenes.
type OrdenFiltro struct {
	Departamento string
	Riesgo       *bool
	Limit, Skip  int
}

// Repo es la capa persistente.
type Repo interface {
	Ping(ctx context.Context) error
	UpsertUsuario(ctx context.Context, u Usuario) error
	GetUsuario(ctx context.Context, nombre string) (*Usuario, error)

	SaveEntrenamiento(ctx context.Context, e Entrenamiento) error // upsert por ID
	GetEntrenamiento(ctx context.Context, id string) (*Entrenamiento, error)
	ListEntrenamientos(ctx context.Context, limit int) ([]Entrenamiento, error)

	InsertPredicciones(ctx context.Context, p []Prediccion) error
	ListPredicciones(ctx context.Context, limit int, usuario string) ([]Prediccion, error)

	InsertMetricasNodos(ctx context.Context, m []MetricaNodo) error
	ListMetricasNodos(ctx context.Context, limit int) ([]MetricaNodo, error)

	InsertOrdenes(ctx context.Context, o []OrdenMuestra) error
	ListOrdenes(ctx context.Context, f OrdenFiltro) ([]OrdenMuestra, int64, error)

	Conteos(ctx context.Context) (map[string]int64, error)
	Close(ctx context.Context) error
}

// Cache es la capa de caché con expiración.
type Cache interface {
	Ping(ctx context.Context) error
	Get(ctx context.Context, key string) ([]byte, bool, error)
	Set(ctx context.Context, key string, val []byte, ttl time.Duration) error
	Close() error
}
