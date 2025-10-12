use std::process::Command;
use std::{fs, net::TcpListener};
use std::io::{BufRead, BufReader, Error, ErrorKind, Result, Write};

pub struct ExecOutput {
    pub stdout: String,
    pub stderr: String,
}

/// Append the text contents passed in to the file pointed at by `filepath`
pub fn append_to_file(filepath: &str, contents: &str) -> Result<()> {
    fs::OpenOptions::new()
        .append(true)
        .open(filepath)
        .map(|mut file| write!(file, "{}", contents))?
}

pub fn call_exec(path: &str, args: &[&String]) -> Result<ExecOutput> {
    let mut command = Command::new(path);
    for arg in args {
        command.arg(arg);
    };
    let output = command.output()?;
    let parsed_output = ExecOutput {
        stdout: String::from_utf8(output.stdout)
            .map_err(|e| Error::new(ErrorKind::Other, e))?,
        stderr: String::from_utf8(output.stderr)
            .map_err(|e| Error::new(ErrorKind::Other, e))?,
    };
    Ok(parsed_output)
}

pub fn get_filenames_in_directory(path: &str) -> Result<Vec<String>> {
    let read_dir = fs::read_dir(path)?;
    let filenames: Vec<String> = read_dir.map(|entry| {
        let name = entry?
            .file_name()
            .into_string()
            .map_err(|_| Error::last_os_error())?;
        Ok(name)
    }).collect::<Result<_>>()?;

    Ok(filenames)
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
