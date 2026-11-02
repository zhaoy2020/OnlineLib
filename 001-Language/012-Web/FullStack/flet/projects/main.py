import flet as ft


class TodoApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "待办事项"
        self.page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        
        # 新任务输入框
        self.new_task = ft.TextField(hint_text="输入新任务", width=300, on_submit=self.add_task)
        # 任务列表
        self.tasks = ft.Column()
        
        # 主界面布局
        self.page.add(
            ft.Text("我的待办", size=24, weight=ft.FontWeight.BOLD),
            ft.Row([self.new_task, ft.ElevatedButton("添加", on_click=self.add_task)]),
            self.tasks
        )

    def add_task(self, e):
        task_text = self.new_task.value.strip()
        if not task_text:
            return
        
        # 创建一个任务行（复选框 + 文字 + 删除按钮）
        def delete_click(e):
            self.tasks.controls.remove(task_row)
            self.tasks.update()
        
        checkbox = ft.Checkbox(value=False, label=task_text)
        delete_btn = ft.IconButton(icon=ft.icons.DELETE, on_click=delete_click)
        task_row = ft.Row([checkbox, delete_btn], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
        
        self.tasks.controls.append(task_row)
        self.new_task.value = ""
        self.page.update()
        self.new_task.focus()


def main(page: ft.Page):
    TodoApp(page)


ft.app(target=main)