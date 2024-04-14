package nnfs

import (
	"archive/zip"
	"fmt"
	"image"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

func Init() {
	//skip all this if the folder exists
	_, err := os.Stat("fashion_mnist_images")
	if err == nil {
		fmt.Println("You already have the file")
		return
	}

	url := "https://nnfs.io/datasets/fashion_mnist_images.zip"
	resp, err := http.Get(url)

	if err != nil {
		fmt.Println("Error on download the data set")
		return
	}

	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		fmt.Println("Error on download the data set:", resp.StatusCode)
		return
	}

	out, err := os.Create("fashion_mnist_images.zip")
	if err != nil {
		fmt.Println("Can't create the file")
		return
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)

	if err != nil {
		fmt.Println("Can't move the files into the zip")
		return
	}

	err = unzip("fashion_mnist_images.zip", "fashion_mnist_images")
	if err != nil {
		fmt.Println("Sorry couldn't unzip, do it manually :D", err)
	}

}

func unzip(src, dest string) error {
	r, err := zip.OpenReader(src)
	if err != nil {
		return err
	}
	defer r.Close()

	for _, f := range r.File {
		rc, err := f.Open()
		if err != nil {
			return err
		}
		defer rc.Close()

		fpath := filepath.Join(dest, f.Name)
		if f.FileInfo().IsDir() {
			os.MkdirAll(fpath, f.Mode())
		} else {
			var fdir string
			if lastIndex := strings.LastIndex(fpath, string(os.PathSeparator)); lastIndex > -1 {
				fdir = fpath[:lastIndex]
			}

			err = os.MkdirAll(fdir, f.Mode())
			if err != nil {
				log.Fatal(err)
				return err
			}
			f, err := os.OpenFile(
				fpath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, f.Mode())
			if err != nil {
				return err
			}
			defer f.Close()

			_, err = io.Copy(f, rc)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func Create_fashion_data(is_test bool) ([]*mat.Dense, *mat.Dense) {

	directory := "test"
	if !is_test {
		directory = "train"
	}

	var X = []*mat.Dense{}
	label_raw := []float64{}

	dirs, _ := os.ReadDir(filepath.Join("fashion_mnist_images", directory))
	for _, dir := range dirs {
		label := dir.Name()
		photos, _ := os.ReadDir(filepath.Join("fashion_mnist_images", directory, label))

		for _, photo := range photos {
			val, _ := strconv.ParseFloat(label, 64)
			label_raw = append(label_raw, val)
			image_file, _ := os.Open(filepath.Join("fashion_mnist_images", directory, label, photo.Name()))
			defer image_file.Close()
			img_raw_data, _, _ := image.Decode(image_file)
			x := mat.NewDense(28, 28, imageToRGB(img_raw_data))
			X = append(X, x)
		}
	}
	var y = mat.NewDense(len(label_raw), 1, label_raw)
	return X, y
}

func Scale_fashion_data(X []*mat.Dense) {
	for _, data := range X {
		data.Apply(func(r, c int, v float64) float64 {
			return (v - 127.5) / 127.5
		}, data)
	}
}

func Reshape_fashion_data(X []*mat.Dense) []*mat.Dense {
	var result = []*mat.Dense{}
	for _, data := range X {
		temp := mat.NewDense(28*28, 1, data.RawMatrix().Data)
		result = append(result, temp)
	}
	return result
}

func imageToRGB(img image.Image) []float64 {
	sz := img.Bounds()
	raw := make([]float64, 28*28)
	idx := 0
	for y := sz.Min.Y; y < sz.Max.Y; y++ {
		for x := sz.Min.X; x < sz.Max.X; x++ {
			r, _, _, _ := img.At(x, y).RGBA()
			raw[idx] = float64(uint8(r))
			idx += 1
		}
	}
	return raw
}
