from . import shakespeare
def words():
    return [w for w in shakespeare.words() if len(w) < 6]
