package main

import (
	"errors"
	"fmt"
	"github.com/Unknwon/goconfig"
	"log"
	"os"
	"time"
	"path"
)

func main() {
	fmt.Println(path.Join("a","b","c"))

	// 获取工程目录
	currentPath, _ := os.Getwd()
	confPath := currentPath + "/init.ini"

	_, err := os.Stat(confPath)
	if err != nil {
		panic(errors.New(fmt.Sprintf("file is not found %s", confPath)))

	}
	// 加载配置
	config, err := goconfig.LoadConfigFile(confPath)
	if err != nil {
		log.Fatal("读取配置文件出错:", err)
	}

	// 获取 section
	mysqlConf, _ := config.GetSection("mysql")
	// map[db_name:blog host:127.0.0.1 max_conn:10 max_idle:2 password:root port:3306 user:root]
	fmt.Println(mysqlConf)
	// 127.0.0.1
	fmt.Println(mysqlConf["host"])

	// 获取单个值
	user, _ := config.GetValue("mysql", "user")
	fmt.Println(user)

	// 获取单个值并且指定类型
	maxIdle, _ := config.Int("mysql", "max_idle")
	fmt.Println(maxIdle)

	// 获取单个值，发生错误时返回默认值，没有默认值返回零值
	port := config.MustInt("mysql", "port", 3308)
	fmt.Println(port)

	// 重新设置值
	config.SetValue("mysql", "port", "3307")
	port = config.MustInt("mysql", "port", 3308)
	fmt.Println(port)

	// 删除值
	config.DeleteKey("mysql", "port")
	port = config.MustInt("mysql", "port", 3308)
	fmt.Println(port)

	// 获取注释
	comments := config.GetKeyComments("mysql", "user")
	fmt.Println(comments)

	// 获取数组，需要指定分隔符
	array := config.MustValueArray("array", "course", ",")
	fmt.Println(array)

	// 重新加载配置文件，一般对于web项目，改了配置文件希望能够即使生效而不需要重启应用，可以对外提供刷新配置api
	// 修改password 为 root123值观察值的变化
	for i := 0; i < 10; i++ {
		time.Sleep(time.Second * 3)
		_ = config.Reload()
		password, _ := config.GetValue("mysql", "password")
		fmt.Println(password)
	}
}
