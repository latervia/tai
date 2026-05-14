import posixpath

if __name__ == '__main__':
    a = "http://100.122.27.37:11434"
    b= posixpath.join(a, "v1")
    print(type(b))

    print(b)
