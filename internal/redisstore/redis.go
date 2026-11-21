// Package redisstore implementa store.Cache sobre Redis (github.com/redis/go-redis/v9).
package redisstore

import (
	"context"
	"errors"
	"time"

	"github.com/redis/go-redis/v9"
)

// Cache es el adaptador de Redis.
type Cache struct{ c *redis.Client }

// New conecta (sin bloquear: los errores de red se tratan en cada operación) a addr "host:puerto".
func New(addr, password string) *Cache {
	return &Cache{c: redis.NewClient(&redis.Options{
		Addr: addr, Password: password,
		DialTimeout: 2 * time.Second, ReadTimeout: 200 * time.Millisecond, WriteTimeout: 200 * time.Millisecond,
		PoolSize: 32, MaxRetries: 1,
	})}
}

func (r *Cache) Ping(ctx context.Context) error { return r.c.Ping(ctx).Err() }
func (r *Cache) Close() error                   { return r.c.Close() }

func (r *Cache) Get(ctx context.Context, key string) ([]byte, bool, error) {
	b, err := r.c.Get(ctx, key).Bytes()
	if errors.Is(err, redis.Nil) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, err
	}
	return b, true, nil
}

func (r *Cache) Set(ctx context.Context, key string, val []byte, ttl time.Duration) error {
	return r.c.Set(ctx, key, val, ttl).Err()
}
