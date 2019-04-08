class CrossSectionReader:

    @staticmethod
    def parse(infile_path):
        try:
            with open(infile_path, 'r') as infile:
                header_line = None
                body_lines = []

                for line in infile:
                    if line.startswith('#'):
                        header_line = line
                    else:
                        body_lines.append(line)
        except:
            print("file '{}' not found or problem reading it!".format(infile_path))

        # now extract the column names
        header_line = header_line.replace('#', '')
        column_names = header_line.split()
        retdict = {column_name: [] for column_name in column_names}

        # then parse the body lines and fill in their values
        for body_line in body_lines:
            contents = body_line.split()
            for content, container in zip(contents, retdict.values()):
                container.append(content)

        # at the end, go through it and prune away empty lists
        for column_name in retdict.keys():
            cur_entry = retdict[column_name]
            if len(cur_entry) == 1:
                retdict[column_name] = cur_entry[0]
            elif len(cur_entry) == 0:
                retdict[column_name] = ''

        return retdict
