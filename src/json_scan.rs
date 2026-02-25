#[inline]
pub(crate) fn skip_ws(bytes: &[u8], mut i: usize) -> usize {
    let len = bytes.len();
    while i < len {
        match bytes[i] {
            b' ' | b'\n' | b'\r' | b'\t' => i += 1,
            _ => break,
        }
    }
    i
}

#[inline]
pub(crate) fn parse_json_string_end(bytes: &[u8], start: usize) -> Result<usize, ()> {
    let len = bytes.len();
    if start >= len || bytes[start] != b'"' {
        return Err(());
    }
    let mut i = start + 1;
    while i < len {
        match bytes[i] {
            b'"' => return Ok(i + 1),
            b'\\' => {
                i += 1;
                if i >= len {
                    return Err(());
                }
                i += 1;
            }
            0x00..=0x1F => return Err(()),
            _ => i += 1,
        }
    }
    Err(())
}

#[inline]
pub(crate) fn parse_json_value_end(bytes: &[u8], start: usize) -> Result<usize, ()> {
    let i = skip_ws(bytes, start);
    if i >= bytes.len() {
        return Err(());
    }

    match bytes[i] {
        b'"' => parse_json_string_end(bytes, i),
        b'{' => parse_json_object_end(bytes, i),
        b'[' => parse_json_array_end(bytes, i),
        b't' => consume_literal(bytes, i, b"true"),
        b'f' => consume_literal(bytes, i, b"false"),
        b'n' => consume_literal(bytes, i, b"null"),
        b'-' | b'0'..=b'9' => parse_json_number_end(bytes, i),
        _ => Err(()),
    }
}

#[inline]
fn parse_json_object_end(bytes: &[u8], start: usize) -> Result<usize, ()> {
    let len = bytes.len();
    if start >= len || bytes[start] != b'{' {
        return Err(());
    }
    let mut i = start + 1;
    loop {
        i = skip_ws(bytes, i);
        if i >= len {
            return Err(());
        }
        match bytes[i] {
            b'}' => return Ok(i + 1),
            b'"' => {}
            _ => return Err(()),
        }

        i = parse_json_string_end(bytes, i)?;
        i = skip_ws(bytes, i);
        if i >= len || bytes[i] != b':' {
            return Err(());
        }
        i = parse_json_value_end(bytes, i + 1)?;
        i = skip_ws(bytes, i);
        if i >= len {
            return Err(());
        }
        match bytes[i] {
            b',' => i += 1,
            b'}' => return Ok(i + 1),
            _ => return Err(()),
        }
    }
}

#[inline]
fn parse_json_array_end(bytes: &[u8], start: usize) -> Result<usize, ()> {
    let len = bytes.len();
    if start >= len || bytes[start] != b'[' {
        return Err(());
    }
    let mut i = start + 1;
    loop {
        i = skip_ws(bytes, i);
        if i >= len {
            return Err(());
        }
        if bytes[i] == b']' {
            return Ok(i + 1);
        }

        i = parse_json_value_end(bytes, i)?;
        i = skip_ws(bytes, i);
        if i >= len {
            return Err(());
        }
        match bytes[i] {
            b',' => i += 1,
            b']' => return Ok(i + 1),
            _ => return Err(()),
        }
    }
}

#[inline]
fn consume_literal(bytes: &[u8], start: usize, lit: &[u8]) -> Result<usize, ()> {
    let end = start.checked_add(lit.len()).ok_or(())?;
    if end <= bytes.len() && &bytes[start..end] == lit {
        Ok(end)
    } else {
        Err(())
    }
}

#[inline]
fn parse_json_number_end(bytes: &[u8], start: usize) -> Result<usize, ()> {
    let len = bytes.len();
    let mut i = start;
    if i < len && bytes[i] == b'-' {
        i += 1;
    }

    if i >= len {
        return Err(());
    }
    match bytes[i] {
        b'0' => i += 1,
        b'1'..=b'9' => {
            i += 1;
            while i < len && bytes[i].is_ascii_digit() {
                i += 1;
            }
        }
        _ => return Err(()),
    }

    if i < len && bytes[i] == b'.' {
        i += 1;
        if i >= len || !bytes[i].is_ascii_digit() {
            return Err(());
        }
        while i < len && bytes[i].is_ascii_digit() {
            i += 1;
        }
    }

    if i < len && matches!(bytes[i], b'e' | b'E') {
        i += 1;
        if i < len && matches!(bytes[i], b'+' | b'-') {
            i += 1;
        }
        if i >= len || !bytes[i].is_ascii_digit() {
            return Err(());
        }
        while i < len && bytes[i].is_ascii_digit() {
            i += 1;
        }
    }

    Ok(i)
}

pub(crate) fn find_top_level_field_value_range(
    bytes: &[u8],
    field_name: &[u8],
) -> Result<Option<std::ops::Range<usize>>, ()> {
    let mut i = skip_ws(bytes, 0);
    if bytes.get(i) != Some(&b'{') {
        return Err(());
    }
    i += 1;

    let mut last_match: Option<std::ops::Range<usize>> = None;
    loop {
        i = skip_ws(bytes, i);
        match bytes.get(i) {
            Some(b'}') => return Ok(last_match),
            Some(b'"') => {}
            _ => return Err(()),
        }

        let key_start = i + 1;
        let key_end = parse_json_string_end(bytes, i)?;
        let key = &bytes[key_start..key_end - 1];

        i = skip_ws(bytes, key_end);
        if bytes.get(i) != Some(&b':') {
            return Err(());
        }
        i = skip_ws(bytes, i + 1);

        let value_start = i;
        let value_end = parse_json_value_end(bytes, i)?;
        if key == field_name {
            last_match = Some(value_start..value_end);
        }
        i = skip_ws(bytes, value_end);

        match bytes.get(i) {
            Some(b',') => i += 1,
            Some(b'}') => return Ok(last_match),
            _ => return Err(()),
        }
    }
}
