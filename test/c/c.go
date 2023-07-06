package main

import (
	"bytes"
	"fmt"
	"net/http"
	"log"
	"io/ioutil"
)

func main() {
	url := fmt.Sprintf("http://%s:%d/post/", "127.0.0.1", 9000)
	resp, err := http.Post(url, "text/plain", bytes.NewBuffer([]byte("ok")))
	if err != nil {
		log.Printf("send complete error:%v", err)
	}
	defer resp.Body.Close()
	data, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		log.Printf("send complete error:%v", err)
	}
	log.Printf("send complete :%s", string(data))
}
