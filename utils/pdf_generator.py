from fpdf import FPDF
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class PDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__curr_image_id = 0
        self.__curr_table_id = 0

    def draw_title(self):
        # Arial bold 15
        self.set_font('Arial', 'B', 16)
        # Calculate width of title and position
        self.set_x(10)
        # Colors of frame, background and text
        self.set_draw_color(255, 255, 255)
        self.set_fill_color(255, 255, 255)
        self.set_text_color(50, 50, 50)
        # Thickness of frame (1 mm)
        self.set_line_width(1)
        # Title
        self.cell(190, 9, self.title, 1, 1, 'C', 1)
        # Line break
        self.ln(10)

    def header(self):
        if self.page_no() > 1:
            # Arial bold 15
            self.set_font('Arial', 'I', 12)
            # Calculate width of title and position
            self.set_x(10)
            # Colors of frame, background and text
            self.set_draw_color(255, 255, 255)
            self.set_fill_color(255, 255, 255)
            self.set_text_color(128)
            # Thickness of frame (1 mm)
            #self.set_line_width(1)
            # Title
            self.cell(190, 9, self.title, 1, 1, 'C', 1)
            # Line break
            self.ln(10)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Text color in gray
        self.set_fill_color(255, 255, 255)
        self.set_text_color(128)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')

    def section_title(self, num, label):
        self.set_text_color(0)
        self.set_fill_color(200, 200, 255)
        # Arial 12
        self.set_font('Arial', 'B', 12)
        # Title
        self.cell(0, 6, 'Section %d : %s' % (num, label), 0, 1, 'L', 1)
        # Line break
        self.ln(4)
        
    def section_body(self, txt):
        # Times 12
        self.set_text_color(0)
        self.set_font('Arial', '', 12)
        # Output justified text
        self.multi_cell(0, 5, txt)
        # Line break
        self.ln()

    def add_table(self, table, title=None):
        self.set_text_color(0)
        if title is not None:
            self.set_font('Times', '', 12)
            # Calculate width of title and position
            w = self.get_string_width(title) + 6
            self.set_x((210 - w) / 2)
            # Colors of frame, background and text
            self.set_text_color(0)
            # Thickness of frame (1 mm)
            self.cell(w, 7, "Table {}: {}".format(self._curr_table_id, title), 0, 0, 'C', 0)
            self.ln()

        self.set_draw_color(0, 0, 0)
        self.set_line_width(0.5)
        n = len(table[0])
        columns_width = [0] * n

        h=10
        for line in table:
            for j, cell in enumerate(line):
                columns_width[j] = max(columns_width[j], self.get_string_width(cell))
        
        col_sum = sum(columns_width)
        margin = min((col_sum-20)/(2*n), 40)
        x = (210 - col_sum)/2 - margin*n
        y = self.get_y()

        for i, line in enumerate(table):
            if i == 0:
                self.set_font('Times', 'B', 12)
                self.set_fill_color(250, 130, 130)
            elif i%2:
                self.set_font('Times', '', 12)
                self.set_fill_color(205)
            else:
                self.set_font('Times', '', 12)
                self.set_fill_color(240)

            self.set_x(x)
            for j, cell in enumerate(line):
                self.cell(columns_width[j]+margin*2, h, cell, 1, 0, 'C', 1)
            self.ln()

        self.ln()

    @property
    def _curr_image_id(self):
        self.__curr_image_id += 1
        return self.__curr_image_id
    
    @property
    def _curr_table_id(self):
        self.__curr_table_id += 1
        return self.__curr_table_id

    def add_images(self, array, titles=[], w=1, force_5_width=False):
        n = len(array)
        if force_5_width:
            n = 5
        # if more than 5 images use more than one row
        if n > 5:
            if len(array) == len(titles):
                for x in range(0, n, 5):
                    self.add_images([array[x + i] for i in range(5) if len(array) > (x+i)],
                                    [titles[x + i] for i in range(5) if len(titles) > (x+i)],
                                        force_5_width=True)
            else:
                for x in range(0, n, 5):
                    self.add_images([array[x + i] for i in range(5) if len(array) > (x+i)],
                                        force_5_width=True)

            return

        c = 1-w
        w = (190 * w - 10 * (n-1)) / n
        h = 0

        for i, img in enumerate(array):
            h=max(h, mpimg.imread(img).shape[0] * w / mpimg.imread(img).shape[1])
        if 287-self.get_y() < h + 5:  
            self.add_page()
        
        y = self.get_y()
        
        for i, img in enumerate(array):
            x = 10 + 10 * i + w * i + (c*190) / 2
            self.image(img, x=x, y=y, w=w)
            h=max(h, mpimg.imread(img).shape[0] * w / mpimg.imread(img).shape[1])
        h = int(h+1)
        y = y+h
        self.set_y(y)
        self.ln

        self.set_font('Times', '', 12)
        for i, t in enumerate(titles):
            if t != "" and t is not None:
                x = 10 + 10 * i + w * i + (c*190) / 2
                self.set_x(x)
                #self.cell(w, 5, "Image {}: {}".format(self._curr_image_id, t), 0, 0, 'C', 0)
                self.cell(w, 5, "{}".format(t), 0, 0, 'C', 0)
        self.ln()
        self.ln()
        self.ln()


if __name__ == "__main__":
    table = [["Model", "Accuracy", "Balanced Accuracy", "FID"],
            ["Resnet", "0.718%", "not done yet", "30"],
            ["VGG", "0.708%", "not done yet", "30"],
            ["VGG", "0.708%", "not done yet", "30"],
            ["VGG", "0.708%", "not done yet", "30"]]
    pdf = PDF()
    pdf.set_title("Title")
    pdf.add_page()
    pdf.draw_title()
    pdf.add_images(['utils/spoty2.jpg'] * 3, ["a", "b", "c"], w=0.5)
    pdf.section_title(1, "Parameters")
    pdf.section_body("I did this!")
    pdf.section_body("And this also")
    pdf.add_table(table, "what if this was a table?")
    pdf.section_title(2, "This is a second section")
    pdf.section_body("DAMN is a great album")
    pdf.add_page()
    pdf.output('tuto3.pdf', 'F')