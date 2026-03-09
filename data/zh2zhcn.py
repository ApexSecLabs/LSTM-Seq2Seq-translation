import zhconv

# 文件路径
input_file = 'cmn.txt'      # 原始文件名
output_file = 'cmn_new.txt'  # 输出文件名

def convert_line(line):
    """处理单行：提取中文部分，繁转简，再组合"""
    line = line.strip()
    if not line:          # 跳过空行
        return ''

    parts = line.split('\t')
    if len(parts) < 3:
        # 如果行格式异常（少于三列），保留原行
        return line

    en = parts[0]                 # 英文句子
    zh = parts[1]                 # 中文句子（可能含繁体）
    attr = '\t'.join(parts[2:])   # 属性信息（可能有多个制表符）

    # 繁体 → 简体 转换
    zh_simple = zhconv.convert(zh, 'zh-cn')

    # 重新组合，仍用制表符分隔
    return f"{en}\t{zh_simple}\t{attr}"

def main():
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            converted = convert_line(line)
            if converted:
                outfile.write(converted + '\n')
            else:
                outfile.write('\n')   # 保留原始空行

    print(f"✅ 转换完成！结果已保存至 {output_file}")

if __name__ == '__main__':
    main()