from handler import Handler


if __name__ == '__main__':
    handler = Handler()
    name = input("?")
    res = handler.get_similar_books(name)

    print(res)
