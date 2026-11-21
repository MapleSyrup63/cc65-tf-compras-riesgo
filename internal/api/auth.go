package api

import (
	"context"
	"crypto/hmac"
	"crypto/pbkdf2"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/grupo4-cc65/cc65-tf-compras-riesgo/internal/store"
)

// JWT HS256 implementado con la biblioteca estándar (crypto/hmac + encoding/base64).

type claims struct {
	Sub string `json:"sub"`
	Rol string `json:"rol"`
	Iat int64  `json:"iat"`
	Exp int64  `json:"exp"`
}

var b64 = base64.RawURLEncoding

const jwtHeader = `{"alg":"HS256","typ":"JWT"}`

func (s *Server) signJWT(user, rol string) (string, time.Time) {
	now := time.Now()
	exp := now.Add(s.cfg.TokenTTL)
	payload, _ := json.Marshal(claims{Sub: user, Rol: rol, Iat: now.Unix(), Exp: exp.Unix()})
	unsigned := b64.EncodeToString([]byte(jwtHeader)) + "." + b64.EncodeToString(payload)
	mac := hmac.New(sha256.New, s.cfg.JWTSecret)
	mac.Write([]byte(unsigned))
	return unsigned + "." + b64.EncodeToString(mac.Sum(nil)), exp
}

var errToken = errors.New("token inválido o expirado")

func (s *Server) parseJWT(tok string) (*claims, error) {
	parts := strings.Split(tok, ".")
	if len(parts) != 3 {
		return nil, errToken
	}
	hdr, err := b64.DecodeString(parts[0])
	if err != nil || string(hdr) != jwtHeader { // fija el algoritmo: evita "alg":"none"
		return nil, errToken
	}
	mac := hmac.New(sha256.New, s.cfg.JWTSecret)
	mac.Write([]byte(parts[0] + "." + parts[1]))
	sig, err := b64.DecodeString(parts[2])
	if err != nil || !hmac.Equal(sig, mac.Sum(nil)) {
		return nil, errToken
	}
	raw, err := b64.DecodeString(parts[1])
	if err != nil {
		return nil, errToken
	}
	var c claims
	if err := json.Unmarshal(raw, &c); err != nil || time.Now().Unix() >= c.Exp {
		return nil, errToken
	}
	return &c, nil
}

// Contraseñas: PBKDF2-HMAC-SHA256 con sal aleatoria (nunca se guarda la contraseña).

const pbkdf2Iter = 120000

func hashPassword(pw string, salt []byte) string {
	k, _ := pbkdf2.Key(sha256.New, pw, salt, pbkdf2Iter, 32)
	return hex.EncodeToString(k)
}

// NewUsuario crea un usuario con sal aleatoria.
func NewUsuario(nombre, password, rol string) store.Usuario {
	salt := make([]byte, 16)
	rand.Read(salt)
	return store.Usuario{Usuario: nombre, Salt: hex.EncodeToString(salt), Hash: hashPassword(password, salt),
		Rol: rol, Creado: time.Now()}
}

func checkPassword(u *store.Usuario, pw string) bool {
	salt, err := hex.DecodeString(u.Salt)
	if err != nil {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(hashPassword(pw, salt)), []byte(u.Hash)) == 1
}

// dummyUser iguala el tiempo de respuesta cuando el usuario no existe (anti enumeración).
var dummyUser = NewUsuario("-", "-", "analista")

type ctxKey struct{}

// authn exige un JWT válido y, si se indican roles, que el rol del token esté entre ellos.
func (s *Server) authn(next http.HandlerFunc, roles ...string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		h := r.Header.Get("Authorization")
		tok, ok := strings.CutPrefix(h, "Bearer ")
		if !ok {
			writeErr(w, http.StatusUnauthorized, "falta el encabezado Authorization: Bearer <token>")
			return
		}
		c, err := s.parseJWT(tok)
		if err != nil {
			writeErr(w, http.StatusUnauthorized, err.Error())
			return
		}
		if len(roles) > 0 {
			allowed := false
			for _, ro := range roles {
				allowed = allowed || ro == c.Rol
			}
			if !allowed {
				writeErr(w, http.StatusForbidden, "permiso insuficiente para este recurso")
				return
			}
		}
		next(w, r.WithContext(context.WithValue(r.Context(), ctxKey{}, c)))
	}
}

func userFrom(r *http.Request) *claims {
	c, _ := r.Context().Value(ctxKey{}).(*claims)
	if c == nil {
		return &claims{}
	}
	return c
}

// rateLimiter: ventana deslizante por IP (protege /auth/login de fuerza bruta).
type rateLimiter struct {
	mu     sync.Mutex
	hits   map[string][]time.Time
	max    int
	window time.Duration
}

func newRateLimiter(max int, window time.Duration) *rateLimiter {
	return &rateLimiter{hits: map[string][]time.Time{}, max: max, window: window}
}

func (l *rateLimiter) allow(key string) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	now := time.Now()
	cut := now.Add(-l.window)
	h := l.hits[key][:0]
	for _, t := range l.hits[key] {
		if t.After(cut) {
			h = append(h, t)
		}
	}
	if len(h) >= l.max {
		l.hits[key] = h
		return false
	}
	l.hits[key] = append(h, now)
	return true
}

func clientIP(r *http.Request) string {
	h, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return h
}
