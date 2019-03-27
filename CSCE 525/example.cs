	
	
	
	string status = "";
	string ccnum = "None";
	try {
		SqlConnection sql= new SqlConnection(
		@"data source=localhost;" +
		"user id=sa;password=pAs$w0rd;");
		sql.Open();
		string sqlstring="SELECT ccnum" +
		" FROM cust WHERE id=" + Id;
		SqlCommand cmd = new SqlCommand(sqlstring,sql);
		ccnum = (string)cmd.ExecuteScalar();
		} catch (SqlException se) {
		status = sqlstring + " failed\n\r";
		foreach (SqlError e in se.Errors) {
		status += e.Message + "\n\r";
	}