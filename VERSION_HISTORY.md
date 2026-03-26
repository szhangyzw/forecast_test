# VERSION_HISTORY

用于记录 `sales_forecast` 项目的版本变更、核心改动、回滚点和备注。

## 版本记录表

| 序号 | Commit ID | Tag | 类型 | 时间 | 说明 | 基于版本 | 是否推荐回滚点 | 备注 |
|---|---|---|---|---|---|---|---|---|
| 1 | be67b91 | v0.1-server-baseline | baseline | 2026-03-26 | 服务器初始基线版本，完成 Git 初始化 | - | 是 | 当前第一版可回滚基线 |
| 2 | e919721 | - | feature | 2026-03-26 | 支持 backtest 多品牌逗号传参；空数据时报错信息更清晰 | be67b91 | 是 | 已包含基线全部内容 |
| 3 | b5bc9b6 | - | feature | 2026-03-26 | run_forecast 支持多品牌；输出全部模型结果并增加推荐标签 | e919721 | 是 | 已包含多品牌 backtest 能力 |

## 建议维护规则

每次重要改动后，补一行：
- `Commit ID`：对应 git commit short hash
- `Tag`：如果打了 tag 就写，没有就填 `-`
- `类型`：如 baseline / feature / fix / refactor / docs
- `时间`：改动完成日期
- `说明`：一句话描述核心变化
- `基于版本`：本次提交直接基于哪个 commit
- `是否推荐回滚点`：是 / 否
- `备注`：风险、兼容性、是否已验证等

## 常用命令

查看提交历史：
```bash
git log --oneline --graph
```

查看当前版本表：
```bash
cat VERSION_HISTORY.md
```

回到某个版本查看：
```bash
git checkout <commit_id>
```

强制回滚到某个版本：
```bash
git reset --hard <commit_id>
```
