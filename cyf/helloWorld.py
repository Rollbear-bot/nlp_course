from handler import Handler


if __name__ == '__main__':
    handler = Handler()

    while True:
        name = input("?")
        res = handler.get_similar_book_lt(name)

        for book in res:
            print(book)
