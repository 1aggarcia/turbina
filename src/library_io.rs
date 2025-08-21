use std::{fs, net::TcpListener};
use std::io::{BufRead, BufReader, Result, Write};

/// Append the text contents passed in to the file pointed at by `filepath`
pub fn append_to_file(filepath: &str, contents: &str) -> Result<()> {
    fs::OpenOptions::new()
        .append(true)
        .open(filepath)
        .map(|mut file| write!(file, "{}", contents))?
}

/// Start a TCP server listening on `address`. Whenever a request is received,
/// it is converted to a string and passed to `handle_request` which should
/// produce a response. The response is sent back to the client.
/// 
/// This function stays running as long as the server is listening.
pub fn open_tcp_server<F>(address: &str, mut handle_request: F) -> Result<()>
where
    F: FnMut(String) -> String
{
    let listener = TcpListener::bind(address)?;
    println!("Server listening on address: {}", address);

    for incoming_stream in listener.incoming() {
        let mut stream = incoming_stream?;
        let tcp_request = BufReader::new(&stream)
            .lines()
            .map(|result| result.unwrap_or("".into()))
            .take_while(|line| !line.is_empty())
            .collect::<Vec<String>>()
            .join("\n");

        let response = handle_request(tcp_request);
        stream.write_all(response.as_bytes())?;
    };
    Ok(())
}
