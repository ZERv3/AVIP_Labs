import ttkbootstrap as ttk

from src.lab_app import ImageLabApp


def main() -> None:
    root = ttk.Window(themename="darkly")
    ImageLabApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
