package nnfs

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

type Model struct {
	Layers     []ILayer
	InputLayer ILayer
	Loss       ILoss
	Optimizer  IOptimizer
	Data_Loss  float64
	Accuracy   IAccuracy
}

func NewModel() *Model {
	return &Model{Layers: []ILayer{}}
}

func (model *Model) Add(layer ILayer) {
	model.Layers = append(model.Layers, layer)
}

func (model *Model) Set(loss ILoss, optimizer IOptimizer) {
	model.Loss = loss
	model.Optimizer = optimizer
}

func (model *Model) Train(X, y *mat.Dense, epochs, print_every int) {
	if print_every <= 0 {
		panic("print_every can't be 0 or negative, make it 1 bro")
	}
	for i := 1; i <= epochs; i++ {
		output := model.Forward(X, y)
		model.Backward(output, y)

		model.Optimizer.PreUpdateParams()

		for _, layer := range model.Layers {
			//TODO: make this better
			switch v := layer.(type) {
			case *LayerDense:
				model.Optimizer.UpdateParameters(v)
			}
		}
		if i%print_every == 0 {
			regularization_loss := 0.0
			for _, layer := range model.Layers {
				//TODO: also make this better
				switch v := layer.(type) {
				case *LayerDense:
					regularization_loss += RegularizationLoss(model.Loss, v)
				}
			}
			loss := model.Data_Loss + regularization_loss
			accuracy := model.Accuracy.Compare(output, y)
			fmt.Println("epoch", i, "data loss -->", model.Data_Loss, "regularization_loss -->", regularization_loss, "loss -->", loss, "lr -->", model.Optimizer.GetLearningRate(), "accuracy -->", accuracy, "%")
		}
		model.Optimizer.PostUpdateParams()
	}
}

func (model *Model) Train_image(X []*mat.Dense, y []*mat.Dense, epochs, batch_size, print_every int) {
	if print_every <= 0 {
		panic("print_every can't be 0 or negative, make it 1 bro")
	}

	if batch_size < 0 {
		panic("batch_size can't be 0 or negative, make it 1 bro")
	}

	steps := len(X) / batch_size
	if steps*batch_size < len(X) {
		steps += 1
	}
	for epoch := 1; epoch <= epochs; epoch++ {
		for step := 0; step < steps; step++ {
			batch_x_raw := []float64{}
			batch_y_raw := []float64{}

			if step > 5 {
				break
			}
			start := step * batch_size
			end := (step + 1) * batch_size
			if end > len(X) {
				end = len(X)
			}
			for i, x := range X[start:end] {
				batch_x_raw = append(batch_x_raw, x.RawMatrix().Data...)
				batch_y_raw = append(batch_y_raw, y[i].RawMatrix().Data...)
			}

			batch_x := mat.NewDense(end-start, X[0].RawMatrix().Cols, batch_x_raw)
			batch_y := mat.NewDense(end-start, y[0].RawMatrix().Cols, batch_y_raw)

			output := model.Forward(batch_x, batch_y)
			model.Backward(output, batch_y)

			model.Optimizer.PreUpdateParams()

			for _, layer := range model.Layers {
				switch v := layer.(type) {
				case *LayerDense:
					model.Optimizer.UpdateParameters(v)
				}
			}
			if step%print_every == 0 {
				regularization_loss := 0.0
				for _, layer := range model.Layers {
					switch v := layer.(type) {
					case *LayerDense:
						regularization_loss += RegularizationLoss(model.Loss, v)
					}
				}
				loss := model.Data_Loss + regularization_loss
				accuracy := model.Accuracy.Compare(output, batch_y)
				fmt.Println("step", step, "data loss -->", model.Data_Loss, "regularization_loss -->", regularization_loss, "loss -->", loss, "lr -->", model.Optimizer.GetLearningRate(), "accuracy -->", accuracy, "%")
			}
			model.Optimizer.PostUpdateParams()
		}

	}
}

func (model *Model) Forward(X, y_true *mat.Dense) *mat.Dense {

	model.InputLayer.Forward(X)
	for _, layer := range model.Layers {
		switch v := layer.(type) {
		case *ActivationSoftMaxLossCategoricalCrossEntropy:
			model.Data_Loss = v.ForwardCombined(v.GetPrevious().GetOutput(), y_true)
		default:
			v.Forward(layer.GetPrevious().GetOutput())
		}
	}
	//
	return model.Layers[len(model.Layers)-1].GetOutput()
}

func (model *Model) Backward(output, y *mat.Dense) {
	for i := len(model.Layers) - 1; i >= 0; i-- {

		layer := model.Layers[i]
		switch v := layer.(type) {
		case *ActivationSoftMaxLossCategoricalCrossEntropy:
			v.BackwardCombined(output, y)
		default:
			v.Backward(v.GetNext().GetDInputs())
		}
	}
}

func (model *Model) Finalize() {
	model.InputLayer = NewInputLayer()
	layer_count := len(model.Layers)
	for index, layer := range model.Layers {
		if index == 0 {
			layer.SetPrevious(model.InputLayer)
			layer.SetNext(model.Layers[index+1])
		} else if index < layer_count-1 {
			layer.SetPrevious(model.Layers[index-1])
			layer.SetNext(model.Layers[index+1])
		} else {
			layer.SetPrevious(model.Layers[index-1])
			layer.SetNext(nil)
		}
	}
}
