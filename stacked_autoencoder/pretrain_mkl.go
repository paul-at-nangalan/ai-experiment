//go:build !native
// +build !native

package stacked_autoencoder

import (
	. "gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/blase"
)

func init() {
	Use(blase.Implementation())
}
