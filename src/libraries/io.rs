use std::fs;
use std::io::{Error, ErrorKind, Result, Write};
use std::process::Command;

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
    }
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
    let filenames: Vec<String> = read_dir
        .map(|entry| {
            let name = entry?
                .file_name()
                .into_string()
                .map_err(|_| Error::last_os_error())?;
            Ok(name)
        })
        .collect::<Result<_>>()?;

    Ok(filenames)
}
