package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"regexp"
)

type Config struct {
	Enabled bool   `json:"enabled"`
	Path    string `json:"path"`
	First   struct {
		Second string `json:"second"`
		Inner  string `json:"inner"`
	} `json:"first"`
}

// 去除jsonc文件中的注释
func removeComments(jsonc string) string {
	commentRegex := regexp.MustCompile(`(?m)(?s)//.*?$|/\*.*?\*/`)
	tmp := commentRegex.ReplaceAllString(jsonc, "")
	re := regexp.MustCompile(`(?m)^\s*$[\r\n]*`)
	return re.ReplaceAllString(tmp, "")
}

// 读取jsonc文件
func readJsonc(jsoncFileName string) (string, error) {
	var err error

	currentPath, _ := os.Getwd()
	confPath := path.Join(currentPath, jsoncFileName)
	_, err = os.Stat(confPath)
	if err != nil {
		return "", errors.New(fmt.Sprintf("config file is not found %s", confPath))
	}

	file, _ := os.Open(confPath)
	defer file.Close()

	jsoncData, err := ioutil.ReadAll(file)
	if err != nil {
		return "", errors.New(fmt.Sprintf("read config file error %s", confPath))
	}

	jsonData := removeComments(string(jsoncData))
	return jsonData, nil
}

func main() {
	jsoncFileName := "config.jsonc"
	jsonData, err := readJsonc(jsoncFileName)
	fmt.Println(jsonData)
	if err != nil {
		panic(err)
	}
	config := Config{}
	json.Unmarshal([]byte(jsonData), &config)

	fmt.Println(config)
	fmt.Println(config.First.Inner)
}
