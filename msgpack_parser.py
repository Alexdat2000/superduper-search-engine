def main():
    import msgpack
    import pickle

    file = open(r"C:\Users\Mi\Downloads\search_items_more.msgpack", "rb")
    unpacker = msgpack.Unpacker(file_like=file)

    dt = {}

    for i in unpacker:
        dt[i["item_id"]] = (i["doc_url"], i["img_url"])

    pickle.dump(dt, open("articles.dump", "wb"))
