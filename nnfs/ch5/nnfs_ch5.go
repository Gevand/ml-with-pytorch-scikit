package nnfs

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

func Run1() {
	soft_max := []float64{0.7, 0.1, 0.2}
	target := []float64{1, 0, 0}

	loss := -(math.Log(soft_max[0])*target[0] +
		math.Log(soft_max[1])*target[1] +
		math.Log(soft_max[2])*target[2])

	//since target[0] is 1 and other indices are 0, it simplifies to this
	loss_equivalent := -(math.Log(soft_max[0]))
	fmt.Println(loss, "vs", loss_equivalent)

	b := 5.2
	fmt.Println(math.Log(b))
	fmt.Println(math.Pow(math.E, math.Log(b)))
}

func Run2() {
	fmt.Println("Cross entropy loss across multiple outputs")
	softmax_outputs := [][]float64{
		{.7, .1, .2},
		{.1, .5, .4},
		{.02, .9, .08}}
	class_targets := []int{0, 1, 1}
	for i, class := range class_targets {
		fmt.Println(softmax_outputs[i][class])
	}

	neg_log := mat.NewDense(3, 1, nil)
	for i, class := range class_targets {
		neg_log.Set(i, 0, -(math.Log(softmax_outputs[i][class])))
	}

	fmt.Println(neg_log)
	sum := mat.Sum(neg_log)
	ave_loss := sum / float64(neg_log.RawMatrix().Rows)
	fmt.Println(ave_loss)
}

func Run3() {
	fmt.Println("Cross entropy loss across multiple outputs, both 2d matrices")
	softmax_outputs := []float64{
		.7, .1, .2,
		.1, .5, .4,
		.02, .9, .08}
	class_targets := []float64{
		1, 0, 0,
		0, 1, 0,
		0, 1, 0}
	softmax_outputs_matrix := mat.NewDense(3, 3, softmax_outputs)
	class_targets_matrix_transposed := mat.NewDense(3, 3, class_targets)
	neg_log := mat.NewDense(3, 1, nil)
	softmax_outputs_matrix.MulElem(softmax_outputs_matrix, class_targets_matrix_transposed)

	neg_log.Apply(func(r, c int, v float64) float64 {
		return clip(-math.Log(mat.Sum(softmax_outputs_matrix.RowView(r))), 1e-7, 1-1e-7)
	}, neg_log)

	ave_loss := mat.Sum(neg_log) / float64(neg_log.RawMatrix().Rows)
	fmt.Println(neg_log, ave_loss)

}

func clip(value, left, right float64) float64 {
	if value >= left && value <= right {
		return value
	}
	if value <= left {
		return left
	}
	return right
}
