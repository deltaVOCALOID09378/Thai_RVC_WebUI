import re

def main():
    try:
        with open('Program.py', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        with open('dump_H.txt', 'w', encoding='utf-8') as f:
            for i, l in enumerate(lines):
                if re.search(r'H:', l, re.IGNORECASE):
                    f.write(f"Line {i+1}: {repr(l)}\n")
    except Exception as e:
        with open('dump_H.txt', 'w', encoding='utf-8') as f:
            f.write(f"Error: {e}\n")

if __name__ == "__main__":
    main()
