from soorgeon import io

if __name__ == '__main__':
    s = """
    """
    in_, out = io.find_inputs_and_outputs(s)
    print(f'Inputs: {in_}')
    print(f'Outputs: {out}')
