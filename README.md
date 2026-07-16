# pianfan.github.io

翩帆的个人技术博客，基于 [Jekyll](https://jekyllrb.com/) 构建，托管在 GitHub Pages。

在线访问：<https://pianfan.github.io>

## 功能特性

- 文章列表分页、按标签分类浏览
- 站内搜索（基于 [Simple-Jekyll-Search](https://github.com/christian-fei/Simple-Jekyll-Search)）
- 明暗主题切换
- 文章目录（TOC）侧边栏
- [giscus](https://giscus.app/) 评论系统（基于 GitHub Discussions）
- RSS 订阅、站点地图（`jekyll-feed` / `jekyll-sitemap`）
- Google Analytics 访问统计
- 代码高亮（Rouge）、GFM 风格 Markdown（kramdown）

## 本地开发

需要 Ruby（3.x）和 Bundler。

```bash
# 安装依赖
bundle install

# 启动本地服务，默认地址 http://127.0.0.1:4000
bundle exec jekyll serve
```

## 撰写文章

在 `_posts/` 目录下新建 Markdown 文件，文件名格式为 `YYYY-MM-DD-title.md`，并添加 Front Matter，例如：

```markdown
---
layout: post
title: "这是文章标题"
date: 2026-07-16
tags: [example]
toc: false
comments: true
author: 张三
---

文章摘要部分……

<!-- more -->

正文内容……
```

- `<!-- more -->` 之前的内容会作为首页摘要显示，如果不加这个标记，则显示正文前约 200 个字符（含结尾的“...”省略号）
- 文章配图建议放在 `images/<文章名>/` 目录下，方便按照文章进行管理

## 目录结构

```text
├── _config.yml          # 站点配置（导航、分页、评论、统计等）
├── _posts/              # 博客文章
├── _layouts/            # 页面布局（default / page / post）
├── _includes/           # 页面组件（导航、页脚、评论、TOC、主题切换等）
├── _sass/               # 样式源文件
├── images/              # 图片资源（头像、favicon、文章配图）
├── js/                  # 前端脚本（站内搜索）
├── tags/                # 标签分类页
├── about.md             # 关于页面
├── index.html           # 首页
├── search.json          # 搜索索引数据源
└── .github/workflows/   # CI：推送到 main 时构建校验站点
```

## 部署

推送到 `main` 分支后由 GitHub Pages 自动构建发布。同时 GitHub Actions（`.github/workflows/build.yml`）会在 push / PR 时执行 `jekyll build` 做构建校验，并上传 `_site` 产物。

## 许可证

代码部分采用 [MIT License](LICENSE)，文章内容版权归作者所有。
