from tkinter import Tk, Label, Text, Button, Scrollbar, END
from tkinter import ttk
from googletrans import Translator, LANGUAGES

class TranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lang translate")

        self.translator = Translator()

        # Input Label and Textbox
        self.input_label = Label(root, text="Enter text to translate:")
        self.input_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.input_textbox = Text(root, height=10, width=50)
        self.input_textbox.grid(row=1, column=0, padx=10, pady=10)

        # Output Label and Textbox
        self.output_label = Label(root, text="Translated text:")
        self.output_label.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        self.output_textbox = Text(root, height=10, width=50)
        self.output_textbox.grid(row=1, column=1, padx=10, pady=10)

        # Scrollbars for Textboxes
        self.scrollbar_input = Scrollbar(root, command=self.input_textbox.yview)
        self.scrollbar_input.grid(row=1, column=0, sticky='nse', pady=10)
        self.input_textbox['yscrollcommand'] = self.scrollbar_input.set

        self.scrollbar_output = Scrollbar(root, command=self.output_textbox.yview)
        self.scrollbar_output.grid(row=1, column=1, sticky='nse', pady=10)
        self.output_textbox['yscrollcommand'] = self.scrollbar_output.set

        # Language Selection Dropdown
        self.languages = LANGUAGES
        self.language_names = list(self.languages.values())
        self.destination_lang_label = Label(root, text="Destination Language:")
        self.destination_lang_label.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        self.destination_lang_combobox = ttk.Combobox(root, values=self.language_names)
        self.destination_lang_combobox.grid(row=2, column=1, padx=10, pady=10)
        self.destination_lang_combobox.set('English')  # Default language is English

        # Buttons
        self.translate_button = Button(root, text="Translate", command=self.translate_text)
        self.translate_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

        self.clear_button = Button(root, text="Clear", command=self.clear_text)
        self.clear_button.grid(row=3, column=1, padx=90, pady=10)

    def translate_text(self):
        input_text = self.input_textbox.get("1.0", END).strip()
        if input_text:
            lang_code = self.get_lang_code(self.destination_lang_combobox.get())
            translation = self.translator.translate(input_text, dest=lang_code)
            self.output_textbox.delete("1.0", END)
            self.output_textbox.insert(END, translation.text)

    def clear_text(self):
        self.input_textbox.delete("1.0", END)
        self.output_textbox.delete("1.0", END)

    def get_lang_code(self, language_name):
        for code, name in self.languages.items():
            if name == language_name:
                return code
        return 'en'  # Default to English if not found

if __name__ == "__main__":
    root = Tk()
    app = TranslatorApp(root)
    root.mainloop()
