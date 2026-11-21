package xlsx

import (
	"archive/zip"
	"os"
	"path/filepath"
	"testing"
)

func writeXLSX(t *testing.T, path string) {
	f, _ := os.Create(path)
	defer f.Close()
	z := zip.NewWriter(f)
	ss, _ := z.Create("xl/sharedStrings.xml")
	ss.Write([]byte(`<sst><si><t>NOMBRE</t></si><si><t>MONTO</t></si><si><t xml:space="preserve">A &amp; B</t></si><si><r><t>RICO</t></r><r><t>TEXTO</t></r></si></sst>`))
	sh, _ := z.Create("xl/worksheets/sheet1.xml")
	sh.Write([]byte(`<worksheet><sheetData>` +
		`<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>` +
		`<row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2"><v>12.5</v></c></row>` +
		`<row r="3"><c r="B3"><v>7</v></c></row>` +
		`<row r="4"><c r="A4" t="s"><v>3</v></c><c r="B4" t="inlineStr"><is><t>99</t></is></c></row>` +
		`</sheetData></worksheet>`))
	z.Close()
}

func TestReadFile(t *testing.T) {
	p := filepath.Join(t.TempDir(), "t.xlsx")
	writeXLSX(t, p)
	for _, w := range []int{1, 2, 8} { // mismo resultado con cualquier número de workers
		sh, err := ReadFile(p, w)
		if err != nil {
			t.Fatal(err)
		}
		if sh.Header[0] != "NOMBRE" || len(sh.Rows) != 3 {
			t.Fatalf("w=%d header/filas incorrectos: %v %d", w, sh.Header, len(sh.Rows))
		}
		if sh.Rows[0][0] != "A & B" || sh.Rows[0][1] != "12.5" {
			t.Errorf("w=%d fila 1: %v", w, sh.Rows[0])
		}
		if sh.Rows[1][0] != "" || sh.Rows[1][1] != "7" {
			t.Errorf("w=%d celda vacía mal resuelta: %v", w, sh.Rows[1])
		}
		if sh.Rows[2][0] != "RICOTEXTO" || sh.Rows[2][1] != "99" {
			t.Errorf("w=%d rich text / inlineStr: %v", w, sh.Rows[2])
		}
	}
}

func TestSplitAt(t *testing.T) {
	data := []byte("<row>1</row><row>2</row><row>3</row><row>4</row>")
	chunks := splitAt(data, rowOpen, 3)
	total := 0
	for _, c := range chunks {
		total += len(c)
		if len(c) > 0 && string(c[:4]) != "<row" {
			t.Errorf("corte fuera de límite de fila: %q", c)
		}
	}
	if total != len(data) {
		t.Errorf("se perdieron bytes al dividir")
	}
}
