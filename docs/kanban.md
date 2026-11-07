# Tablero Kanban (GitHub Projects) — tarjetas sugeridas

Columnas: **Backlog · To Do · In Progress · Review · Done**

## Done (PC3)
| # | Tarjeta | Rama | Responsable |
|---|---|---|---|
| 1 | Definir caso, pregunta de impacto social y etiqueta de riesgo | docs/planteamiento | Integrante 1 |
| 2 | Lector XLSX concurrente en Go puro | feature/lector-xlsx | Integrante 1 |
| 3 | Limpieza y validación (RUC, montos, fechas, codificación, moneda) | feature/etl-limpieza | Integrante 2 |
| 4 | Pipeline productor → workers → agregador con deduplicación | feature/etl-limpieza | Integrante 2 |
| 5 | Ingeniería de variables y etiqueta (map-reduce, target encoding LOO) | feature/features-etiqueta | Integrante 3 |
| 6 | EDA concurrente + figuras | feature/eda | Integrante 3 |
| 7 | Random Forest distribuido (sharding + bagging) | feature/random-forest | Integrante 1 |
| 8 | Regresión logística con SGD sincrónico data-parallel | feature/logreg | Integrante 2 |
| 9 | Pruebas unitarias, `-race`, benchmarks de speedup | feature/pruebas-benchmarks | Integrante 3 |
| 10 | Dockerfile + docker-compose | feature/docker | Integrante 1 |
| 11 | Informe PC3 + reporte de participación | docs/informe-pc3 | Todos |

## Backlog (PC4)
- Nodo ML en Go con servidor TCP (protocolo JSON/gob) — reutiliza `TrainTreesOnShard` y `LocalGradient`
- API coordinadora REST + WebSocket + JWT, Swagger
- MongoDB (predicciones, métricas) y Redis (caché de scores)
- Pruebas de latencia con Postman (< 100 ms)

## Backlog (TB2)
- SPA React: login, dashboard de impacto, simulador de orden, panel del clúster
- Video de 6 minutos
