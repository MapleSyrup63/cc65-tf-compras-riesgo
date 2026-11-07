package etl

import "testing"

func fila(monto, estado, moneda, freg, femi, contratista string) []string {
	r := make([]string, NumCols)
	r[ColEntidad] = "MUNICIPALIDAD DISTRITAL DE PRUEBA"
	r[ColRucEntidad] = "20123456789"
	r[ColDepartamento] = "PROV.CONSTITUC.DEL CALLAO"
	r[ColTipoOrden] = "Orden de Compra"
	r[ColNroOrden] = "12"
	r[ColOrden] = "O/C-12-2026"
	r[ColDescripcion] = "ADQUISICION DE COMBUSTIBLE DIESEL"
	r[ColObjeto] = "BIENES"
	r[ColEstado] = estado
	r[ColTipoContratacion] = "Contrataciones hasta 8 UIT (No incluye las derivadas de contrataciones por catálogo electrónico)"
	r[ColMonto] = monto
	r[ColMoneda] = moneda
	r[ColRucContratista] = "10456789012"
	r[ColNombreContratista] = contratista
	r[ColFechaRegistro] = freg
	r[ColFechaEmision] = femi
	return r
}

func TestCleanRowValida(t *testing.T) {
	motivo, r := CleanRow(fila("1000", "Anulada", "Dólar Norteamericano", "2026-03-10", "2026-03-01", "PE¿A  LOPEZ "), "MARZO")
	if r == nil {
		t.Fatalf("esperaba registro válido, motivo=%s", motivo)
	}
	if r.Departamento != "CALLAO" {
		t.Errorf("departamento = %q", r.Departamento)
	}
	if r.Contratista != "PEÑA LOPEZ" {
		t.Errorf("corrección de codificación falló: %q", r.Contratista)
	}
	if r.MontoPEN != 3700 {
		t.Errorf("conversión a soles: %v", r.MontoPEN)
	}
	if !r.Anulada || r.TipoContratacion != "HASTA_8UIT" || r.Rubro != "COMBUSTIBLE" || r.TipoContratista != "PN" {
		t.Errorf("campos derivados incorrectos: %+v", r)
	}
	if r.NivelGobierno != "MUNI_DISTRITAL" {
		t.Errorf("nivel = %s", r.NivelGobierno)
	}
}

func TestCleanRowRechazos(t *testing.T) {
	casos := map[string][]string{
		RejMonto:       fila("0", "Emitida", "Nuevos Soles", "2026-03-10", "2026-03-01", "X"),
		RejOrdenFechas: fila("10", "Emitida", "Nuevos Soles", "2026-02-10", "2026-03-01", "X"),
		RejFecha:       fila("10", "Emitida", "Nuevos Soles", "10/03/2026", "2026-03-01", "X"),
		RejEstado:      fila("10", "Desconocido", "Nuevos Soles", "2026-03-10", "2026-03-01", "X"),
	}
	for esperado, f := range casos {
		if motivo, r := CleanRow(f, "MARZO"); r != nil || motivo != esperado {
			t.Errorf("esperaba %s, obtuve %s", esperado, motivo)
		}
	}
	bad := fila("10", "Emitida", "Nuevos Soles", "2026-03-10", "2026-03-01", "X")
	bad[ColRucEntidad] = "20ABC"
	if motivo, _ := CleanRow(bad, "M"); motivo != RejRucEntidad {
		t.Errorf("RUC inválido no detectado: %s", motivo)
	}
}

func TestRubro(t *testing.T) {
	casos := map[string]string{
		"SERVICIO DE CONSULTORIA PARA ELABORACION DE EXPEDIENTE": "CONSULTORIA",
		"CONTRATACION DE ASISTENTE ADMINISTRATIVO":               "PERSONAL",
		"ADQUISICIÓN DE VÍVERES PARA COMEDOR":                    "ALIMENTOS",
		"XYZ":                                                    "OTROS",
	}
	for desc, esp := range casos {
		if got := ClasificarRubro(normText(desc)); got != esp {
			t.Errorf("%q -> %s, esperaba %s", desc, got, esp)
		}
	}
}

func TestNewerDeterministico(t *testing.T) {
	_, a := CleanRow(fila("10", "Comprometida", "Nuevos Soles", "2026-03-10", "2026-03-01", "X"), "MARZO")
	_, b := CleanRow(fila("10", "Anulada", "Nuevos Soles", "2026-04-10", "2026-03-01", "X"), "ABRIL")
	if !newer(b, a) || newer(a, b) {
		t.Error("debe preferir el archivo más reciente")
	}
	_, c := CleanRow(fila("10", "Anulada", "Nuevos Soles", "2026-03-10", "2026-03-01", "X"), "MARZO")
	if !newer(c, a) {
		t.Error("con el mismo mes debe preferir el estado más avanzado")
	}
}
