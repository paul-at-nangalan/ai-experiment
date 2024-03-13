//go:build native
// +build native

package stacked_autoencoder

import (
	"gonum.org/v1/gonum/blas/gonum"
	. "gorgonia.org/gorgonia"
)

func init() {
	Use(gonum.Implementation{})
}
