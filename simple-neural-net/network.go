package simple_neural_net

import (
	"gonum.org/v1/gonum/mat"
	"math"
	"math/rand"
	"time"
)

/// Adapted from https://datadan.io/blog/neural-net-with-go

type NeuralNetConfig struct {
	InputNeurons    int
	OutputNeurons   int
	HiddenLayers    int
	HiddenNeurons   int
	HiddenLayerCols int
	HiddenLayerRows int
	NumEpochs       int
	LearningRate    float64
}

type NeuralNet struct {
	config  NeuralNetConfig
	wHidden []*mat.Dense
	bHidden []*mat.Dense
	wOut    *mat.Dense
	bOut    *mat.Dense

	randgen *rand.Rand
}

func createMatrix(numlayers, rows, cols int, randgen *rand.Rand) []*mat.Dense {
	matrix := make([]*mat.Dense, numlayers)
	for i, _ := range matrix {
		matrix[i] = mat.NewDense(rows, cols, nil)
		rawdata := matrix[i].RawMatrix().Data
		for i := range rawdata {
			rawdata[i] = randgen.Float64()
		}
	}
	return matrix
}

// sigmoid implements the sigmoid function
// for use in activation functions.
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// sigmoidPrime implements the derivative
// of the sigmoid function for backpropagation.
func sigmoidPrime(x float64) float64 {
	return sigmoid(x) * (1.0 - sigmoid(x))
}

func NewNeuralNet(cfg NeuralNetConfig) *NeuralNet {
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	return &NeuralNet{
		config:  cfg,
		wHidden: createMatrix(cfg.HiddenLayers, cfg.InputNeurons, cfg.HiddenNeurons, randGen),
		bHidden: createMatrix(cfg.HiddenLayers, 1, cfg.HiddenNeurons, randGen),
		wOut:    createMatrix(1, cfg.InputNeurons, cfg.HiddenNeurons, randGen)[0],
		bOut:    createMatrix(1, 1, cfg.HiddenNeurons, randGen)[0],
		randgen: randGen,
	}
}

func (p *NeuralNet) fwdLayer(from, wto, bto *mat.Dense) {
	wto.Mul(from, wto) /// is this safe

	addBHidden := func(_, col int, v float64) float64 { return v + bto.At(0, col) }
	wto.Apply(addBHidden, wto)
	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	wto.Apply(applySigmoid, wto)

}

func (p *NeuralNet) fwd(input *mat.Dense) {
	p.fwdLayer(input, p.wHidden[0], p.bHidden[0])
	i := 1
	for ; i < len(p.wHidden); i++ {
		p.fwdLayer(p.wHidden[i-1], p.wHidden[i], p.bHidden[i])
	}
	p.fwdLayer(p.wHidden[i], p.wOut, p.bOut)
}

func (p *NeuralNet) backLayer(from, wto, bto *mat.Dense) {

}

func (p *NeuralNet) back(output *mat.Dense) {

}

func (p *NeuralNet) Train() {
	output := new(mat.Dense)

}

func (p *NeuralNet) Backpropogate(x, youtput *mat.Dense) {
	for i := 0; i < p.config.NumEpochs; i++ {
		p.bHidden[0].Mul()
	}
}
