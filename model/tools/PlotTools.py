from PIL import Image, ImageDraw, ImageFont

class PlotTools():

    @staticmethod
    def save_layers_for_plot(model):
        denses = model.layers
        table_data = []
        for i in range(len(denses)):
            dense = denses[i]

            if dense.prev_layer.__class__.__name__ == "InputLayer":
                layer_input = []
                layer_output = []

                layer_input.append('dense_input')
                layer_input.append("input:")
                layer_input.append([None, dense.prev_layer.get_outs_number()])

                layer_output.append(dense.prev_layer.__class__.__name__)
                layer_output.append('output:')
                layer_output.append([None, dense.prev_layer.get_outs_number()])

                table_data.append(layer_input)
                table_data.append(layer_output)

            layer_input = []
            layer_output = []

            layer_input.append('dense' if i == 0 else f'dense_{i}')
            layer_input.append("input:")
            layer_input.append([None, dense.prev_layer.get_outs_number()])

            layer_output.append(dense.__class__.__name__)
            layer_output.append('output:')
            layer_output.append([None, dense.get_outs_number()])

            table_data.append(layer_input)
            table_data.append(layer_output)
        return table_data

    @classmethod
    def plot_model(cls, model, filename):
        image = Image.new("RGB", (0, 0), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        table_data = cls.save_layers_for_plot(model)

        max_width = sum(
            [max([draw.textsize(str(row[i]), font=font)[0] + 10 for row in table_data]) for i in
             range(len(table_data[0]))])
        max_height = len(table_data) * 20 + ((len(table_data) // 2) - 1) * 40

        # Создаем новое изображение
        width, height = max_width + 20, max_height + 20
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)

        # max_len = 0
        start_x, all_x, prev_all_x = 0, 0, 0
        x, y = 0, 10
        cell_height = 20
        line_height = 20
        for i in range(0, len(table_data), 2):
            rows = table_data[i:i + 2]
            if not len(rows) == 0:
                column_widths = [max([draw.textsize(str(row[i]), font=font)[0] + 10 for row in rows]) for i in
                                 range(len(rows[0]))]
                all_x = sum(column_widths)
                # max_len = all_x if all_x > max_len else max_len
                if start_x == 0:
                    x = (max_width - all_x) / 2 + (width - max_width) / 2
                if not prev_all_x == 0:
                    center = start_local_x + (prev_all_x / 2)
                    draw.line(((center, y), (center, y + line_height)), fill="black", width=1)
                    draw.polygon(
                        [(center - 7, y + line_height), (center + 7, y + line_height), (center, y + line_height * 2)],
                        fill='black')
                    y += line_height * 2
                    x = start_local_x + (prev_all_x - all_x) / 2
                start_local_x = x
                for row in rows:
                    for i, cell in enumerate(row):
                        draw.rectangle([x, y, x + column_widths[i], y + cell_height], outline="black")
                        draw.text((x + 5, y + 5), str(cell), font=font, fill="black")
                        x += column_widths[i]
                    y += cell_height
                    x = start_local_x
                prev_all_x = all_x
        image.save(filename)
        # image.show()

