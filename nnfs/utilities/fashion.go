package nnfs

import (
	"archive/zip"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

func Init() {
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

	unzip("fashion_mnist_images.zip", "fashion_mnist_images")

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
