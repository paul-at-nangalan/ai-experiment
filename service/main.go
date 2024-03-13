package service

//// Copied and slightly mod'd from the gorgonia examples

import "C"

import (
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"runtime/pprof"

	autoenc "ai-experiment/stacked_autoencoder"
	"gonum.org/v1/gonum/blas/gonum"
	T "gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/examples/mnist"
	"gorgonia.org/tensor"
)

// /Convert the following to a struct
type StackedDASettings struct {
	Cpuprofile       string //= flag.String("cpuprofile", "", "write cpu profile to file")
	Memprofile       string // = flag.String("memprofile", "", "write memory profile to this file")
	Dataset          string // = flag.String("dataset", "dev", "which data set to train on? Valid options: \"train\" or \"dev\"")
	Batchsize        int    // = flag.Int("bs", 1, "training batch size (doesn't affect sgd batch size)")
	PretrainingEpoch int    // = flag.Int("pt", 16, "pretraining epoch")
	FinetuningEpoch  int    // = flag.Int("ft", 40, "finetuning epoch")
	Viz              int    // = flag.Int("viz", 0, "Visualize which layer?")
	Saveas           string // = flag.String("save", "", "Save file as")
	Verbose          bool   // = flag.Bool("v", false, "Verbose?")
	Location         string
}

type StackedDAService struct {
	trainingWriter io.Writer
	trainingLog    *log.Logger

	settings StackedDASettings
}

var dt tensor.Dtype = tensor.Float64

func NewStackedDAService(sdasettings StackedDASettings) *StackedDAService {
	var err error
	svc := &StackedDAService{
		settings: sdasettings,
	}
	if svc.trainingWriter, err = os.OpenFile("training.viz", os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0644); err != nil {
		log.Fatal(err)
	}

	svc.trainingLog = log.New(svc.trainingWriter, "", log.Ltime|log.Lmicroseconds)

	return svc
}

func predictBatch(logprobs tensor.Tensor, batchSize int) (guesses []int, err error) {
	var argmax tensor.Tensor
	if batchSize == 1 {
		argmax, err = tensor.Argmin(logprobs, 0)
	} else {
		argmax, err = tensor.Argmin(logprobs, 1)
	}
	if err != nil {
		return nil, err
	}
	guesses = argmax.Data().([]int)
	return
}

func makeTargets(targets tensor.Tensor) []int {
	ys := make([]int, targets.Shape()[0])
	ys = ys[:0]
	for i := 0; i < targets.Shape()[0]; i++ {
		ysl, _ := targets.Slice(T.S(i))
		raw := ysl.Data().([]float64)
		for i, v := range raw {
			if v == 0.9 {
				ys = append(ys, i)
				break
			}
		}
	}
	return ys
}

func (p *StackedDAService) verboseLog(format string, attrs ...interface{}) {
	if p.settings.Verbose {
		log.Printf(format, attrs...)
	}
}

func (p *StackedDAService) createSda(inputs tensor.Tensor) (*T.ExprGraph, *autoenc.StackedDA) {
	g := T.NewGraph()
	size := inputs.Shape()[0]
	inputSize := 784
	outputSize := 10
	hiddenSizes := []int{1000, 1000, 1000}
	layers := len(hiddenSizes)
	corruptions := []float64{0.1, 0.2, 0.3}
	sda := autoenc.NewStackedDA(g, p.settings.Batchsize, size, inputSize, outputSize, layers, hiddenSizes, corruptions, p.trainingLog, dt)
	return g, sda
}

func (p *StackedDAService) Train() {
	flag.Parse()
	//rand.NewSource(1337)

	trainOn := p.settings.Dataset
	inputs, targets, err := mnist.Load(trainOn, p.settings.Location, dt)
	if err != nil {
		log.Fatal(err)
	}
	g, sda := p.createSda(inputs)

	// start CPU profiling before we start training
	if p.settings.Cpuprofile != "" {
		f, err := os.Create(p.settings.Cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	p.verboseLog("Pretraining...")
	for i := 0; i < p.settings.PretrainingEpoch; i++ {
		if err = sda.Pretrain(inputs, i); err != nil {
			os.WriteFile("fullGraph_err.dot", []byte(g.ToDot()), 0644)
			log.Panicf("i: %d err :%v", i, err)
		}
	}

	// Because for now LispMachine doesn't support batched BLAS
	p.verboseLog("Starting to finetune now")

	T.Use(gonum.Implementation{})
	ys := makeTargets(targets)
	for i := 0; i < p.settings.FinetuningEpoch; i++ {
		if err = sda.Finetune(inputs, ys, i); err != nil {
			log.Panicln(err)
		}
	}

	// save model
	if p.settings.Saveas != "" {
		if err = sda.Save(p.settings.Saveas); err != nil {
			log.Panicln(err)
		}
	}

}

func (p *StackedDAService) Predict() {
	/* PREDICTION TIME */

	// here I'm using the test dataset as prediction.
	// in real life you should probably be doing crossvalidations and whatnots
	// but in this demo, we're going to skip all those
	p.verboseLog("pred")
	/// We need to replace this to load in live data
	testX, testY, err := mnist.Load("test", p.settings.Location, dt)
	if err != nil {
		log.Fatal(err)
	}

	///load the sda
	_, sda := p.createSda(testX)

	var one, correct, lp tensor.Tensor
	if one, err = testX.Slice(T.S(0, p.settings.Batchsize)); err != nil {
		log.Fatal(err)
	}

	if correct, err = testY.Slice(T.S(0, p.settings.Batchsize)); err != nil {
		log.Fatal(err)
	}

	correctYs := makeTargets(correct)

	var predictions []int
	if lp, err = sda.Forwards(one); err != nil {
		log.Fatal(err)
	}

	if predictions, err = predictBatch(lp, p.settings.Batchsize); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Correct: \n%+v. \nPredicted: %v. \nLogprobs: \n%+#3.3s", correctYs, predictions, lp)
}
