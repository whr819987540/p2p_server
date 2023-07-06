package main

import (
	"fmt"
	"log"
	"net/http"
	"io/ioutil"
)

func post(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "invalid", http.StatusMethodNotAllowed)
	} else {
		data, err := ioutil.ReadAll(r.Body)
		if err != nil {
			log.Printf("send complete error:%v", err)
		}
		log.Printf("send complete :%s", string(data))
		log.Print("recv")
		fmt.Fprintf(w, "OK")
		return
	}
}

func main() {
	http.HandleFunc("/post/", post)

	if err := http.ListenAndServe(fmt.Sprintf(":%d", 9000), nil); err != nil {
		log.Printf("listen %d error", 9000)
	}
}
