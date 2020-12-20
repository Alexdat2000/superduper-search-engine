def main():
    import msgpack
    import pickle

    unpacker = msgpack.Unpacker(file_like=open(r"C:\Users\Mi\Downloads\search_items_more.msgpack", "rb"))
    unpacker2 = msgpack.Unpacker(file_like=open(r"Q:\search_items.msgpack", "rb"))

    dt = {}

    for i in unpacker:
        dt[i["item_id"]] = i["doc_url"]

    for i in unpacker2:
        x = dt[i["item_id"]]
        dt[i["item_id"]] = (i["title"], x, i["content"][:120])
    pickle.dump(dt, open("articles.dump", "wb"))


main()
