---
layout: post
title: "零基础小白如何搭建自己的 github.io 个人网站"
date: 2024-01-27
tags: [tutorial]
toc: true
comments: true
author: Pianfan
---

###### 说明：本教程只针对不了解网站搭建并且想要快速搭建起个人博客的新手，帮助建立网站的平台有很多，有一定网站开发基础的读者可另寻门路<!-- more -->

## Step 0. 准备工作

利用 GitHub Pages 免费获取你自己的网站域名只需要一个先决条件：拥有你自己的 github 账号。如果你目前还没有，可以去 [GitHub 官网](https://github.com)先注册一个。

## Step 1. 建立博客仓库

借助 GitHub 平台搭建博客网站，首先要建立一个与你的 github 账号相关联的博客仓库。推荐没有网站建设经验的新手通过 fork 我的博客仓库迅速建立起你的第一个网站。下面我们就来详细介绍这种方法。

点击链接 <https://github.com/pianfan/pianfan.github.io> 进入我的博客仓库地址，点击这里的 Fork。

![点击 Fork](https://pianfan.github.io/images/build_own_website/fork.png)

仓库名称填写 `username.github.io`，注意 `username` 指的是用户名不是昵称。

![设置仓库名称](https://pianfan.github.io/images/build_own_website/repositoryname.png)

点击 Create fork，完成创建。这样你就有了你的第一个博客仓库，**以后的操作都在你自己的仓库进行**。

这个地方也不要忘记改成你自己的网址哦~

![设置网址](https://pianfan.github.io/images/build_own_website/modifywebsite.png)

## Step 2. 修改仓库文件

如果这时你在浏览器网址一栏搜索你刚刚建立的仓库名，出现的会是我的博客页面。因为你刚才是直接复刻的我的博客仓库，里面的文件都是我原来仓库里的文件。你需要修改几个文件里的几个最基本的地方，来使页面与你的个人信息相关联。

- ### 修改 README.md 文件

  README.md 文件是对仓库的说明，一般来说第一行标题都是仓库名称，后面就是关于仓库的一些介绍。我只填写了最简要的说明，你可以根据你自己的情况来写，或者不写也可以，它并不影响我们的网站页面。

  要修改文件可直接点击文件名称，然后点击右上角的笔图案，即可开始编辑。

  ![点击笔](https://pianfan.github.io/images/build_own_website/pen.png)

  .md 文件是 markdown 文件，使用的是 Markdown 语言。这一易学易用的语言将 HTML 里的一些常见标签都用带有特殊含义的符号来表示，大大方便了文档的书写。我们的博客文章也需要用 Markdown 进行写作。

  github 支持对 markdown 文件进行预览。当在 github 上对 markdown 文件进行编辑的时候，点击 Edit 旁边的 Preview，即可预览当前的页面效果。这在编辑过程中经常用来检查文章排版是否符合预期。

  ![切换预览](https://pianfan.github.io/images/build_own_website/preview.png)

- ### 修改 _config.yml 文件

  _config.yml 文件是博客网站的核心配置文件，里面包含了你想要显示在网站上的各种构造信息。下面我会一步一步告诉你哪些信息需要改。

  - **网站名称和网站描述**

    ![名称和描述](https://pianfan.github.io/images/build_own_website/name&desc.png)

    这个根据你自己的喜好来设置，不一定要仿造我的模式。比如你可以给你的博客网站取一个有特色的名字，网站描述也可以是简短的自我介绍或个性签名等任何你想表达的内容。

  - **个人头像和网站 logo**

    ![头像和 logo](https://pianfan.github.io/images/build_own_website/avatar&ico.png)

    avatar 代表头像，后面的链接是你想显示在页面的头像图片的 url。favicon 指网站图标，即显示在浏览器标签页和收藏夹里的 logo，通常以 32 * 32 像素大小的 .ico 图片为宜，也可以不设置。

    咱博客网站里的所有图片不是上传到 github 仓库里就可以显示到页面上了，需要用到图床。我用的是 [PicGo](https://picgo.github.io/PicGo-Doc/zh/)，只要与 github 仓库绑定就可以实现上传，且可以一键复制为 Markdown 形式，方便写文时插入图片。

    ![PicGo 设置](https://pianfan.github.io/images/build_own_website/picgoset.png)

    ![Markdown 形式](https://pianfan.github.io/images/build_own_website/markdownimg.png)

  - **个人社交链接**

    ![社交链接](https://pianfan.github.io/images/build_own_website/links.png)

    填用户名就好，也可以不填。

  - **脚注和网址**

    ![版权标注和网址](https://pianfan.github.io/images/build_own_website/footer&url.png)

  - **Gitalk 配置信息**

    ![Gitalk 配置](https://pianfan.github.io/images/build_own_website/gitalk.png)

    Gitalk 用于给博客文章引入评论功能，配置方法请参考 <https://github.com/gitalk/gitalk?tab=readme-ov-file#usage>。

  - **其他**

    如果你不知道改了之后会有什么后果，不要去动它。

- ### 清空 _posts 文件夹

  _posts 文件夹里放的是博客文章，你以后的文章也要放在这里。现在你的 _posts 文件夹里面放的还是我的文章，请把它们全部删除。直接在 GitHub 上操作似乎不支持批量删除文件，一个快捷的方法是删除整个文件夹再新建一个空文件夹，将其命名为“_posts”即可。

- ### 清空 images 文件夹

  images 文件夹里放的是需要显示在网站上的图片。与 _posts 文件夹一样，这个文件夹也需要清空（删除文件夹并重建）。

- ### 修改 about.md 文件

  about.md 里的内容是展示在“关于”页面上的。

好了，以上就是所有必须要你修改的文件。你现在再点进去你的博客页面看看，不出意外的话应该是成功修改了的。

## Step 3. 开始写你的第一篇文章

在 README.md 部分我有提到过怎么在 github 上编写 Markdown 文件，写文章也是一样的道理。当然，你完全可以在其他地方编辑你的文章源码，再上传到博客仓库，这其实也是我更为推荐的方法。

文章文件名请按照下面的例子呈现的格式命名：

    2024-01-25-letter_to_you.md

还有一点需要注意，每篇文章开头记得附上说明，格式如下：

    ---
    layout: post
    title: "文章标题"
    date: 2024-01-27
    tags: [tag1, tag2]
    toc: true
    comments: false
    author: xxx
    ---

  - `tags` 是文章标签，可以有 0 个或多个。
  - `toc` 用于控制是否开启文章侧栏目录。
  - `comments` 用于控制是否开启文章评论区。

---

本篇教程就到此结束啦！如果有任何问题，欢迎在下方评论区留言~
