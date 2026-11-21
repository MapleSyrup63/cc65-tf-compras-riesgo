// Package etl contiene la Parte A del TF: limpieza, validación y
// transformación concurrente de las órdenes de compra/servicio del Estado (OSCE).
package etl

import "time"

// Columnas del archivo fuente (orden del XLSX publicado en datosabiertos.gob.pe).
const (
	ColEntidad = iota
	ColRucEntidad
	ColDepartamento
	ColTipoOrden
	ColNroOrden
	ColOrden
	ColDescripcion
	ColObjeto
	ColEstado
	ColTipoContratacion
	ColMonto
	ColMoneda
	ColRucContratista
	ColNombreContratista
	ColFechaRegistro
	ColFechaEmision
	ColFechaCompromiso
	ColFechaNotificacion
	NumCols
)

// Header esperado; se valida contra el archivo para detectar cambios de esquema.
var Header = []string{
	"ENTIDAD", "RUC_ENTIDAD", "DEPARTAMENTO_ENTIDAD", "TIPOORDEN", "NRO_DE_ORDEN",
	"ORDEN", "DESCRIPCION_ORDEN", "OBJETOCONTRACTUAL", "ESTADOCONTRATACION",
	"TIPODECONTRATACION", "MONTO_TOTAL_ORDEN_ORIGINAL", "MONEDA", "RUC_CONTRATISTA",
	"NOMBRE_RAZON_CONTRATISTA", "FECHA_REGISTRO", "FECHA_DE_EMISION",
	"FECHA_COMPROMISO_PRESUPUESTAL", "FECHA_DE_NOTIFICACION",
}

// UIT 2026 = S/ 5,500 (D.S. N.° 301-2025-EF). Tope de contratación directa: 8 UIT.
const (
	UIT2026  = 5500.0
	Tope8UIT = 8 * UIT2026
)

// Record es una orden ya limpia y normalizada.
type Record struct {
	Entidad          string
	RucEntidad       string
	Departamento     string
	NivelGobierno    string
	EsOrdenCompra    bool
	NroOrden         int
	Orden            string
	Descripcion      string
	Rubro            string
	Objeto           string
	Estado           string
	TipoContratacion string // categoría agrupada
	MontoOriginal    float64
	Moneda           string
	MontoPEN         float64
	RucContratista   string
	TipoContratista  string // PN (RUC 10), EMPRESA (RUC 20), OTRO
	Contratista      string
	FechaRegistro    time.Time
	FechaEmision     time.Time
	ArchivoMes       string

	// Campos derivados en la etapa de features
	Anulada    bool
	Sobrecosto bool
	Riesgo     bool
	EsTrain    bool
}
