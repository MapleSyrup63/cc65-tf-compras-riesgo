# Predicción de Contrataciones y Riesgo en Compras Públicas (Perú, 2026)

**CC65 – Programación Concurrente y Distribuida · UPC · Trabajo Final · Caso N.° 4**
Grupo 4 · Entregable 1 (PC3)

Sistema en **Go** que carga, limpia y procesa de forma concurrente **2.32 millones de órdenes de compra y servicio del Estado** (OSCE / SEACE, enero–septiembre 2026, Portal de Datos Abiertos del Perú) y entrena en paralelo modelos de clasificación binaria para **detectar órdenes con alto riesgo de sobrecosto o cancelación**.

> **Pregunta de impacto social:** *¿Qué órdenes de compra pública —y en qué entidades y regiones— tienen mayor riesgo de sobrecosto o anulación, para que los órganos de control (Contraloría, OCI, OECE) prioricen su revisión antes de que se comprometa y gaste el dinero público?*

## Resultados PC3

| Indicador | Valor |
|---|---|
| Filas leídas / válidas | 2,324,083 / 2,320,035 |
| Duplicados eliminados / rechazos | 3,970 / 78 |
| Pipeline completo (lectura + limpieza + features + EDA) | ≈ 65 s en 2 núcleos |
| Órdenes en riesgo (etiqueta) | 171,444 (7.39 %) |
| Random Forest (100 árboles, 2 workers) | ROC-AUC 0.956 · PR-AUC 0.833 · F1 0.769 |
| Regresión logística (SGD data-parallel) | ROC-AUC 0.908 · PR-AUC 0.596 · F1 0.582 |
| Revisión del top 10 % (RF) | captura el 82.7 % de las órdenes en riesgo (lift 8.3x) |
| Speedup con 2 workers (2 núcleos) | cargador 1.69x · RF 1.86x · LR 1.89x |
| `go test -race` | 12 pruebas OK, 0 condiciones de carrera |

## Estructura

```
cmd/
  loader/     Parte A: XLSX → limpieza → dedup → features → EDA (concurrente)
  train/      Parte B: Random Forest y Regresión Logística en paralelo + benchmarks
  xlsx2csv/   herramienta auxiliar: convierte los XLSX a CSV crudo
internal/
  xlsx/       lector XLSX en Go puro (zip + XML por bloques en paralelo)
  etl/        limpieza, validación, rubros, etiqueta y features (map-reduce)
  eda/        estadísticas descriptivas (map-reduce)
  ml/         dataset, binning, Random Forest, regresión logística, métricas
reports/      evidencias JSON/TXT (ETL, EDA, entrenamiento, benchmarks, tests)
docs/         diagramas (.dot) e imágenes del informe
data/sample/  muestra de 20,000 órdenes limpias (datos de prueba)
scripts/      figuras del informe (solo dibujan los JSON generados por Go)
```

Sin dependencias externas: solo la librería estándar de Go 1.24.

## Cómo ejecutar

1. Descargar los 9 archivos `ORDENESCOMPRA<MES>2026_0.xlsx` desde datosabiertos.gob.pe (OSCE – Órdenes de compra y servicio) y copiarlos en `data/raw/`.
2. Ejecutar:

```bash
go run ./cmd/loader -in data/raw -out data/processed      # Parte A (~1 min)
go run ./cmd/train  -data data/processed/dataset.bin -trees 100 -depth 16 -epochs 5   # Parte B (~25 s)

# evidencias
go test -race -v ./...
go run ./cmd/loader -in data/raw -bench      # escalabilidad del cargador (1,2,4,8 workers)
go run ./cmd/train  -bench                   # escalabilidad de RF y LR
```

### Docker

```bash
docker compose up --build        # loader y luego trainer
```

`Dockerfile.scratch` es una variante sin descargas (imagen `scratch` con binarios estáticos) usada para la evidencia de la PC3.

## Git Flow

- `main`: versiones entregadas (tags `v0.3.0-pc3`, `v0.4.0-pc4`, `v1.0.0-tb2`)
- `develop`: integración
- `feature/*`: una rama por tarea del tablero Kanban, merge `--no-ff` a `develop`
- `release/*`: estabilización antes de cada entrega

## Próximos entregables

- **PC4:** nodos ML en contenedores comunicados por TCP con la API coordinadora (REST + WebSocket + JWT), MongoDB y Redis, Swagger, latencia < 100 ms.
- **TB2:** SPA (React) con login, dashboard de impacto, simulación de predicciones y panel de métricas del clúster.
