package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/exec"
	"sync"
	"time"
	"io"

	"github.com/gin-gonic/gin"
	"github.com/vladimirvivien/go4vl/device"
)

type Record struct {
	Time   time.Time `json:"time"`
	Status string    `json:"status"`
	Image  []byte    `json:"image"`
}

var records []Record

func main() {
	wg := sync.WaitGroup{}
	r := gin.Default()
	r.GET("/data", func(c *gin.Context) {
		c.JSON(http.StatusOK, records)
	})
	wg.Add(1)
	go func() {
		r.Run()
		wg.Done()
	}()
	dev, err := device.Open("/dev/video0", device.WithBufferSize(1))
	if err != nil {
		log.Fatal(err)
	}
	defer dev.Close()

	if err := dev.Start(context.TODO()); err != nil {
		log.Fatal(err)
	}

	frame := <-dev.GetOutput()

	file, err := os.Create("input.jpg")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	if _, err := file.Write(frame); err != nil {
		log.Fatal(err)
	}

	cmd := exec.Command("./mask", "./input.jpg")
	if err := cmd.Start(); err != nil {
		log.Fatal(err)
	}

	log.Println("processed")

	success := true
	status := "OK"
	if err := cmd.Wait(); err != nil {
		if exiterr, ok := err.(*exec.ExitError); ok {
			_ = exiterr
			success = false
			status = "Mask Violation"
		} else {
			log.Fatal(err)
		}
	}
	_ = success

	f, err := os.Open("output.jpg")
	if err != nil {
		log.Fatal(err)
	}

	image, err := io.ReadAll(f)
	if err != nil {
		log.Fatal(err)
	}

	records = append(records, Record{
		Status: status,
		Image:  image,
		Time:   time.Now(),
	})
	wg.Wait()
}
