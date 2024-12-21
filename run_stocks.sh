tickers=("YINN" "AAPL" "MSFT" "GOOG" "AMZN" "FB" "TSLA" "NVDA" "MRNA" "PFE" "AMGN" "GILD" "BNTX" "BABA" "TCEHY" "PDD" "JD" "AMD" "OXY" "TQQQ" "SOXL" "SOXS" "SQQQ" "OXY" "ROIT" "BILI" "MARA")

# 循环遍历每个股票代码并执行 Python 脚本
for ticker in "${tickers[@]}"; do
    echo "Running script for ticker: $ticker"
    python alpha-factor-RL17.py "$ticker"

    # 检查上一个命令的退出状态
    if [ $? -ne 0 ]; then
        echo "Error occurred while processing ticker: $ticker. Exiting."
        exit 1  # 如果发生错误，退出脚本
    fi
done
