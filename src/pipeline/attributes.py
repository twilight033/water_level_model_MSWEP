"""已废弃：属性构建改由 pipeline.attribute_sources 承担。

本模块原先经 `hydrodataset.Camelsh.read_attr_xrdataset` 读取 config.py 的
`ATTRIBUTE_VARIABLES`。该路径有两个问题：

1. hydrodataset 只暴露一张映射表内的变量子集，名字不在表里就报"不是标准变量
   名"——项目早期据此误判为"部分属性不可用"，实际上 86 个流域在 28 个属性
   文件中均有完整记录，697 个数值属性可用。
2. 它无法表达扩展属性集所需的 (输出名, 源文件, 原始列, 类型) 四元组。

保留本文件只为让误用**立即失败**：若继续沿用旧路径，会生成一份与新管线不一致
的属性表且不报错，进而污染实验结果。
"""

_MESSAGE = (
    "pipeline.attributes 已废弃，请改用 pipeline.attribute_sources：\n"
    "    from pipeline.attribute_sources import get_attribute_table\n"
    "    df, onehot = get_attribute_table('base')      # 或 'extended'\n"
    "属性集在训练中由 TrainConfig.attr_set 选择，不再读 config.ATTRIBUTE_VARIABLES。"
)


def build_attributes(*args, **kwargs):
    raise RuntimeError(_MESSAGE)


if __name__ == "__main__":
    raise SystemExit(_MESSAGE)
