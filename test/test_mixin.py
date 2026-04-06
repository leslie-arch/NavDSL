class LogMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # 必须调用
        self.log = "log"

    def show_super(self):
        super_cls: BaseA = super()
        print(f"super class of MyClass: {type(super_cls)}")
        super_cls.introduce()


class BaseA:
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.name = name

    def introduce(self):
        print("I'm Base class A.")


class BaseB:
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.name = name

    def introduce(self):
        print("I'm Base class B.")

class MyClass(LogMixin, BaseB):
    pass

# 正确初始化
obj = MyClass(name="example")
obj.show_super()
