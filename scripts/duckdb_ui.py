import duckdb

def main():
    c = duckdb.connect('output/beams.duckdb')
    c.execute("INSTALL ui; LOAD ui;")
    c.execute('CALL start_ui()')
    input('UI running at http://localhost:4213 - press Enter to stop...')

if __name__ == '__main__':
    main()
