#!/bin/bash
# ============================================================
# Finsler-Adam GitHub Push Script
# ============================================================
# 使い方:
#   1. このスクリプトの GITHUB_USER を自分のアカウント名に変更
#   2. ターミナルで実行:
#        cd "/Users/kuyu/砂場/☆Finsler　Adam　開発/05_GitHub公開パッケージ"
#        bash github_push.sh
#
# 前提条件:
#   - git がインストール済み
#   - gh (GitHub CLI) がインストール済み (brew install gh)
#   - gh auth login で認証済み
# ============================================================

set -e  # エラーで停止

# ★ ここを自分のアカウント名に変更 ★
GITHUB_USER="tsukuyu-lab"
REPO_NAME="finsler-adam"

echo "============================================"
echo "  Finsler-Adam GitHub Push"
echo "============================================"
echo ""

# Step 1: Git初期化
echo "[1/5] Git を初期化中..."
git init
git branch -M main

# Step 2: .gitignore確認
echo "[2/5] ファイルをステージング中..."
git add .

# Step 3: 初回コミット
echo "[3/5] 初回コミット..."
git commit -m "$(cat <<'EOF'
Initial release: Finsler-Adam v0.1.0

Asymmetric-metric optimizer with critical scaling gradient clipping for PyTorch.

Components:
- FinslerAdam optimizer (drop-in AdamW replacement)
- Anna-Limit: smooth 4/3-exponent gradient clipping
- Finsler Scaling: direction-dependent step sizes

Includes:
- Synthetic benchmark results (5 functions × 4 configs × 3 LRs)
- Hyperparameter robustness analysis (γ × α grid search)
- Theoretical compute cost comparison (6 optimizers)
- Examples: CIFAR-10 ResNet-20, GPT-2 Small, Colab notebook
- 8 unit tests with AdamW equivalence verification
- CI: GitHub Actions (Python 3.8-3.12 × PyTorch 1.13-2.3)
- Technical report (LaTeX, 4 pages)
EOF
)"

# Step 4: GitHub リポジトリ作成 + push
echo "[4/5] GitHub リポジトリを作成中..."

# gh CLI が使える場合
if command -v gh &> /dev/null; then
    echo "  gh CLI を検出。自動作成します..."
    gh repo create "$REPO_NAME" \
        --public \
        --description "Finsler-Adam: Asymmetric-metric optimizer with critical scaling gradient clipping for PyTorch" \
        --homepage "https://github.com/$GITHUB_USER/$REPO_NAME" \
        --source . \
        --push
else
    echo ""
    echo "  ⚠ gh CLI が見つかりません。手動でリポジトリを作成してください:"
    echo ""
    echo "  1. https://github.com/new を開く"
    echo "  2. リポジトリ名: $REPO_NAME"
    echo "  3. Public を選択"
    echo "  4. README/LICENSE/gitignore は追加しない（既にあるため）"
    echo "  5. 'Create repository' をクリック"
    echo ""
    echo "  作成後、以下を実行:"
    echo "    git remote add origin https://github.com/$GITHUB_USER/$REPO_NAME.git"
    echo "    git push -u origin main"
    echo ""
    read -p "  リポジトリを作成したら Enter を押してください... "
    git remote add origin "https://github.com/$GITHUB_USER/$REPO_NAME.git"
    git push -u origin main
fi

# Step 5: 確認
echo ""
echo "[5/5] 完了!"
echo ""
echo "============================================"
echo "  リポジトリURL: https://github.com/$GITHUB_USER/$REPO_NAME"
echo "============================================"
echo ""
echo "次のステップ:"
echo "  1. README が正しく表示されるか確認"
echo "  2. GitHub の Settings > Topics に以下を追加:"
echo "     pytorch, optimizer, deep-learning, adamw, gradient-clipping"
echo "  3. About の Description を設定"
echo "  4. PyPI 公開: python -m build && twine upload dist/*"
echo ""
