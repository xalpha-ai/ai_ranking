# Troubleshooting Guide

## Error: "Unexpected token '<', "<html> <"... is not valid JSON"

This error occurs when the server returns an HTML error page instead of JSON. I've added comprehensive error handling to fix this.

### What Was Fixed:

1. **Global Error Handlers**: All Flask errors now return JSON instead of HTML
2. **Better Logging**: Detailed logs show exactly where errors occur
3. **Non-Blocking Database**: Database save failures won't break the analysis
4. **Frontend Error Detection**: Better error messages in the browser console

### How to Deploy the Fix:

```bash
cd "/Users/xalpha/Documents/AI Ranking"
git add .
git commit -m "Add comprehensive error handling"
git push
```

Railway will auto-deploy in ~2 minutes.

### How to Debug on Railway:

1. **View Logs**:
   - Go to your Railway project
   - Click on your web service
   - Click "Deployments" tab
   - Click on the latest deployment
   - View real-time logs

2. **Look for These Log Messages**:
   ```
   === Generate Report Request ===
   Brand: [brand name], Industry: [industry]
   Running visibility audit...
   ✓ Visibility audit completed
   ✓ Saved visibility score for [brand]
   ✓ Report generated
   ```

3. **Common Errors to Look For**:

   **Database Connection Error**:
   ```
   ⚠ Database initialization warning: [error]
   ```
   **Fix**: Make sure PostgreSQL is added in Railway

   **Import Error**:
   ```
   ModuleNotFoundError: No module named 'database'
   ```
   **Fix**: Ensure database.py is committed to Git

   **OpenAI API Error**:
   ```
   OpenAI API error: [error]
   ```
   **Fix**: Check your API key is valid

### Test Locally First:

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (will use SQLite instead of PostgreSQL)
python app.py

# Test at http://localhost:8080
```

### Check Browser Console:

1. Open browser DevTools (F12)
2. Go to Console tab
3. Run an analysis
4. Look for error messages

If you see:
```
Server returned non-JSON response: <html>...
```

Then check the Railway logs for the actual error.

### Manual Test API:

You can test the API directly:

```bash
curl -X POST https://your-app.railway.app/generate-report \
  -H "Content-Type: application/json" \
  -d '{
    "brand": "Test Brand",
    "industry": "Test Industry",
    "description": "A test brand",
    "api_key": "sk-..."
  }'
```

Should return JSON, not HTML.

### Database Issues:

If database saves are failing (you'll see `⚠ Failed to save to database` in logs):

1. **Check PostgreSQL is running**:
   - Go to Railway project
   - PostgreSQL service should show "Active"

2. **Check DATABASE_URL exists**:
   - Click on web service
   - Go to "Variables" tab
   - Should see `DATABASE_URL` variable

3. **Check database tables exist**:
   - Click on PostgreSQL service
   - Go to "Data" tab
   - Should see tables: `visibility_scores`, `competitive_analysis`, etc.

### Still Getting Errors?

The analysis will still work even if database saves fail! You'll still get:
- ✅ Visibility reports
- ✅ Competitive analysis
- ✅ Ranking analysis
- ✅ Geographic analysis
- ✅ PDF exports

You just won't have:
- ❌ Historical trends in dashboard
- ❌ Brand dropdown populated

The dashboard will show: "No data yet - Run your first analysis"

This is okay for demo purposes. Fix the database later if needed.

## Other Common Errors:

### "All fields are required"
**Cause**: Missing form fields
**Fix**: Fill in all fields: Brand, Industry, Description, API Key

### "OpenAI API error"
**Cause**: Invalid or expired API key
**Fix**: Check your OpenAI API key at https://platform.openai.com/api-keys

### "Rate limit exceeded"
**Cause**: Too many API calls
**Fix**: Wait a few minutes, or upgrade OpenAI plan

### "Timeout error"
**Cause**: Analysis taking too long
**Fix**: Railway might have request timeout. Analysis should still complete in background.

## Support

- Check Railway logs first
- Check browser console second
- Review this troubleshooting guide
- File an issue if problem persists

---

**Most Important**: Even if database saves fail, the core functionality (analysis and reports) will still work!
